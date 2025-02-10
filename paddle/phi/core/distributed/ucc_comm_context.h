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
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/distributed/ucc_tools.h"
#include "paddle/phi/core/distributed/utils.h"

namespace phi {
class DenseTensor;
namespace distributed {

constexpr auto kWaitTimeout = std::chrono::milliseconds(0);
constexpr auto kDefaultTimeout = std::chrono::milliseconds(30 * 60 * 1000);
#define PADDLE_UCC_DEVICE_NOT_SET -2

class UCCComm;

class UCCCommContext final : public CommContext {
 public:
  // C++ class wrapper for ucc_coll_req_t.
  class ProgressEntry {
    friend class UCCCommContext;
    friend class UCCComm;

   public:
    explicit ProgressEntry(ucc_coll_req_h request) : request_(request) {}
    ucc_coll_req_h request_;
  };

  class UCCCommTask {
    friend class UCCCommContext;
    friend class UCCComm;

   public:
    explicit UCCCommTask(uint64_t seq) : seq_(seq) {}
    ~UCCCommTask();
    bool IsCompleted();
    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);

   protected:
    std::shared_ptr<ProgressEntry> entry_;
    uint64_t seq_;
  };

  UCCCommContext(const std::shared_ptr<Store>& store,
                 int rank,
                 int size,
                 std::chrono::duration<float> timeout = kDefaultTimeout);
  ~UCCCommContext() override;

  void InitComm(Place place);

  template <typename PreProcess, typename PostProcess>
  std::shared_ptr<UCCCommTask> CollectivePost(CommType commType,
                                              PreProcess preproc,
                                              PostProcess postproc,
                                              ucc_coll_args_t* coll,
                                              Place place);

  std::shared_ptr<UCCCommTask> Reduce(phi::DenseTensor* out_tensor,
                                      const phi::DenseTensor& in_tensor,
                                      ucc_reduction_op_t reduce_type,
                                      int root,
                                      gpuStream_t stream);

 protected:
  uint64_t seq_{0};
  const std::chrono::duration<float> timeout_;
  std::shared_ptr<PaddleUCCOobCollInfo> oob;
  std::shared_ptr<UCCComm> ucc_comm = {nullptr};
  uint32_t comm_id;
  ucc_team_h team{nullptr};
  ucc_ee_h cuda_ee{nullptr};
  ucc_ee_h cuda_ee_p2p[2]{nullptr, nullptr};

  void SetTimeout(ucc_coll_args_t* args);
};

class UCCComm : public UCCCommBase {
  std::shared_ptr<PaddleUCCOobCollInfo> oob;
  std::mutex mutex;
  std::thread progress_thread;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;
  std::deque<std::shared_ptr<UCCCommContext::UCCCommTask>> progress_queue;
  bool stop_progress_loop;
  bool collective_inprogress;

 public:
  int8_t cuda_device_index;
  UCCComm(std::shared_ptr<PaddleUCCOobCollInfo> oob, Place place);

  ~UCCComm();

  void UCCCreateTeam(ucc_team_h* team,
                     std::shared_ptr<PaddleUCCOobCollInfo> oob);

  void UCCDestroyTeam(ucc_team_h* team);

  static std::shared_ptr<UCCComm> GetComm(
      uint32_t* id, Place place, std::shared_ptr<PaddleUCCOobCollInfo> oob);

  void EnqueueCollective(std::shared_ptr<UCCCommContext::UCCCommTask> task,
                         ucc_coll_args_t* coll,
                         ucc_team_h team);

  void ProgressLoop();
};

}  // namespace distributed
}  // namespace phi
