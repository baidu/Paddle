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

#pragma once

#include <ucc/api/ucc.h>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_with_stream.h"
#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/distributed/ucc_comm_context.h"
#include "paddle/phi/core/distributed/ucc_tools.h"
#include "paddle/phi/core/platform/device_event.h"

namespace paddle {
namespace distributed {

using Place = phi::Place;
using UCCCommTask = phi::distributed::UCCCommContext::UCCCommTask;

class ProcessGroupUCC final : public ProcessGroupWithStream {
 public:
  class UCCTask final : public ProcessGroupWithStream::TaskStream,
                        public std::enable_shared_from_this<UCCTask> {
   public:
    UCCTask(const Place& place,
            int rank,
            CommType comm_type,
            bool sync_op,
            bool use_calc_stream);
    virtual ~UCCTask();

    bool IsCompleted() override;
    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout) override;
    void Synchronize() override;
    void SetTask(std::shared_ptr<UCCCommTask> task);

   private:
    Place task_place_;
    std::shared_ptr<UCCCommTask> task_;
  };

 public:
  static std::shared_ptr<ProcessGroupUCC> CreateProcessGroupUCC(
      const std::shared_ptr<phi::distributed::Store>& store,
      int rank,
      int size,
      int gid);

  ProcessGroupUCC(const std::shared_ptr<phi::distributed::Store>& store,
                  int rank,
                  int size,
                  int gid);

  std::string GetBackendName() const override { return "UCC"; }

  phi::DeviceContext* GetDeviceContext(const Place& place) const override;

  phi::DeviceContext* GetDeviceContext(const Place& place,
                                       bool use_calc_stream) const override;

  //   std::shared_ptr<ProcessGroup::Task> AllGather(
  //       phi::DenseTensor* out_tensor,
  //       const phi::DenseTensor& in_tensor,
  //       int64_t offset,  // for compatibility, no use now
  //       int64_t numel,   // for compatibility, no use now
  //       bool sync_op,
  //       bool use_calc_stream) override;

  //   std::shared_ptr<ProcessGroup::Task> AllReduce(
  //       phi::DenseTensor* out_tensor,
  //       const phi::DenseTensor& in_tensor,
  //       const AllreduceOptions& opts,
  //       bool sync_op,
  //       bool use_calc_stream) override;

  //   std::shared_ptr<ProcessGroup::Task> AllToAll(
  //       phi::DenseTensor* out_tensor,
  //       const phi::DenseTensor& in_tensor,
  //       const std::vector<int64_t>& out_size_each_rank,
  //       const std::vector<int64_t>& in_size_each_rank,
  //       bool sync_op,
  //       bool use_calc_stream) override;

  //   std::shared_ptr<ProcessGroup::Task> Broadcast(
  //       phi::DenseTensor* out_tensor,
  //       const phi::DenseTensor& in_tensor,
  //       const BroadcastOptions& opts,
  //       bool sync_op,
  //       bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(phi::DenseTensor* out_tensor,
                                             const phi::DenseTensor& in_tensor,
                                             const ReduceOptions& opts,
                                             bool sync_op,
                                             bool use_calc_stream) override;

  //   std::shared_ptr<ProcessGroup::Task> ReduceScatter(
  //       phi::DenseTensor* out_tensor,
  //       const phi::DenseTensor& in_tensor,
  //       const ReduceScatterOptions& opts,
  //       bool sync_op,
  //       bool use_calc_stream) override;

  //   std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
  //                                            int src_rank,
  //                                            int64_t offset,
  //                                            int64_t numel,
  //                                            bool sync_op,
  //                                            bool use_calc_stream) override;

  //   std::shared_ptr<ProcessGroup::Task> Send(const phi::DenseTensor& tensor,
  //                                            int dst_rank,
  //                                            int64_t offset,
  //                                            int64_t numel,
  //                                            bool sync_op,
  //                                            bool use_calc_stream) override;

  //   std::shared_ptr<ProcessGroup::Task> Barrier(
  //       const BarrierOptions& = BarrierOptions()) override;

  // static void GroupStart();

  // static void GroupEnd();

  // UCCContext_t UCCComm(const Place& place) const;

  // phi::distributed::UCCCommContext* GetOrCreateCommContext(
  //     const Place& place, CommType comm_type = CommType::UNKNOWN);

 private:
  std::shared_ptr<ProcessGroupUCC::UCCTask> CreateTask(const Place& place,
                                                       int rank,
                                                       CommType op_type,
                                                       bool sync_op,
                                                       bool use_calc_stream);

  void CreateUCCEnvCache(const Place& place, const std::string& place_key);

  phi::distributed::UCCCommContext* GetCommContext(
      const std::string* key = nullptr);

  std::shared_ptr<ProcessGroup::Task> Collective(
      std::function<std::shared_ptr<UCCCommTask>(
          phi::distributed::UCCCommContext*, gpuStream_t)> fn,
      const phi::DenseTensor& tensor,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> Point2Point(
      std::function<void(phi::distributed::UCCCommContext*, gpuStream_t, int)>
          fn,
      int peer,
      const phi::DenseTensor& tensor,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<phi::distributed::Store> store_;

  std::unordered_map<std::string, platform::DeviceEvent>
      place_to_calc_event_;  // event on calc stream
  std::unordered_map<std::string, phi::GPUContext*> place_to_calc_ctx_;
  std::unordered_map<std::string, std::unique_ptr<phi::GPUContext>>
      place_to_comm_ctx_;
};

}  //  namespace distributed
}  //  namespace paddle
