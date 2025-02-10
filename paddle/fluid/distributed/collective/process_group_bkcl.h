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

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_with_stream.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/platform/gen_comm_id_helper.h"

#if defined(PADDLE_WITH_XPU)
#include "paddle/fluid/distributed/collective/bkcl_tools.h"
#endif

constexpr const char* BKCL_BACKEND_NAME = "BKCL";

namespace paddle {
namespace distributed {

using Place = phi::Place;

// BKCL funcs use separate communication stream by default
class ProcessGroupBKCL : public ProcessGroupWithStream {
 public:
  class BKCLTask final : public ProcessGroupWithStream::TaskStream,
                         public std::enable_shared_from_this<BKCLTask> {
   public:
    BKCLTask(const Place& place,
             int rank,
             CommType CommType,
             bool sync_op,
             bool use_calc_stream);
    virtual ~BKCLTask();

    using CommFunc = std::function<void()>;
    // TODO(zhangxiaoci): XPU do not support event query for now
    bool IsCompleted() override;
    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout) override;
    void Synchronize() override;
    void RegisterDelayedCommFunc(const CommFunc& func);

    void SynchronizeStreams();

   public:
    bool barrier_{false};
    std::shared_ptr<XPUEventManager> comm_event_;  // event on comm stream
    std::vector<CommFunc> delayed_funcs_;

   private:
    bool HasDelayedCommFunc() const { return !delayed_funcs_.empty(); }
    void StartDelayedCommFunc();
    Place place_;
  };

 public:
  ProcessGroupBKCL(const std::shared_ptr<phi::distributed::Store>& store,
                   int rank,
                   int size,
                   int gid);

  static std::shared_ptr<ProcessGroupBKCL> CreateProcessGroupBKCL(
      const std::shared_ptr<phi::distributed::Store>& store,
      int rank,
      int size,
      int gid);

  std::string GetBackendName() const override {
    return std::string(BKCL_BACKEND_NAME);
  }

  phi::DeviceContext* GetDeviceContext(const Place& place) const override;

  phi::DeviceContext* GetDeviceContext(const Place& place,
                                       bool use_calc_stream) const override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t offset,  // for compatibility, no use now
      int64_t numel,   // for compatibility, no use now
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const std::vector<int64_t>& out_size_each_rank,
      const std::vector<int64_t>& in_size_each_rank,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(phi::DenseTensor* out_tensor,
                                             const phi::DenseTensor& in_tensor,
                                             const ReduceOptions& opts,
                                             bool sync_op,
                                             bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ReduceScatterOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                           int src_rank,
                                           int64_t offset,
                                           int64_t numel,
                                           bool sync_op,
                                           bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Send(const phi::DenseTensor& tensor,
                                           int dst_rank,
                                           int64_t offset,
                                           int64_t numel,
                                           bool sync_op,
                                           bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  static void GroupStart();

  static void GroupEnd();

  BKCLContext_t BKCLComm(const Place& place) const;

  phi::distributed::BKCLCommContext* GetOrCreateCommContext(
      const Place& place, CommType comm_type = CommType::UNKNOWN);

 private:
  std::shared_ptr<ProcessGroupBKCL::BKCLTask> CreateTask(const Place& place,
                                                         int rank,
                                                         CommType op_type,
                                                         bool sync_op,
                                                         bool use_calc_stream);

  void BroadcastUniqueBKCLID(BKCLUniqueId* bkcl_id);

  void CreateBKCLEnvCache(const Place& place, const std::string& place_key);

  std::shared_ptr<ProcessGroup::Task> Collective(
      std::function<void(phi::distributed::BKCLCommContext*, XPUStream)> fn,
      const phi::DenseTensor& tensor,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> Point2Point(
      std::function<void(phi::distributed::BKCLCommContext*, XPUStream, int)>
          fn,
      int peer,
      const phi::DenseTensor& tensor,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  void SyncCalcStream(const Place& place);
  phi::distributed::BKCLCommContext* GetCommContext();

  virtual void StartCoalescing();

  virtual void EndCoalescing(
      std::optional<std::vector<std::shared_ptr<ProcessGroup::Task>>>
          tasks_opt = std::nullopt);

 private:
  std::shared_ptr<phi::distributed::Store> store_;
  std::mutex mutex_;
  std::shared_ptr<XPUEventManager> calc_event_;  // event on calc stream
  std::unordered_map<std::string, phi::XPUContext*> place_to_calc_ctx_;
  std::unordered_map<std::string, std::unique_ptr<phi::XPUContext>>
      place_to_comm_ctx_;

  // For colaescing tensors processing (eg. batch_isend_irecv)
  bool is_coalescing_{false};
  std::vector<std::string> colaescing_place_keys_;
};

}  //  namespace distributed
}  //  namespace paddle
