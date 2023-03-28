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

#pragma once

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"
#include "paddle/fluid/distributed/fleet_executor/task_loop_thread_pool.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
class Scope;
class ProgramDesc;
}  // namespace framework

namespace distributed {

class TaskNode;
class InterceptorMessageServiceImpl;
class RuntimeGraph;
class MessageBus;

class Carrier final {
 public:
  explicit Carrier(int32_t carrier_id) : carrier_id_(carrier_id) {}
  ~Carrier();
  void Init(
      int64_t rank,
      const std::unordered_map<int64_t, int64_t>& interceptor_id_to_rank,
      const std::unordered_map<int64_t, TaskNode*>& interceptor_id_to_node,
      const framework::ProgramDesc& program,
      framework::Scope* scope,
      framework::Scope* minibatch_scope,
      const platform::Place& place,
      const std::vector<framework::Scope*>& micro_scope_list,
      TaskLoopThreadPool* thread_pool);

  void Release();
  void ClearMicroScopes();

  // Enqueue a message to corresponding interceptor id
  bool EnqueueInterceptorMessage(const InterceptorMessage& interceptor_message);

  // get interceptor based on the interceptor id
  Interceptor* GetInterceptor(int64_t interceptor_id);

  // set interceptor with interceptor id
  Interceptor* SetInterceptor(int64_t interceptor_id,
                              std::unique_ptr<Interceptor>);

  void SetSourceInterceptor(Interceptor* interceptor) {
    source_interceptor_ = interceptor;
  }
  void SetSinkInterceptor(Interceptor* interceptor) {
    sink_interceptor_ = interceptor;
  }

  void Start();

  bool IsInit() const;

  bool Send(const InterceptorMessage& msg);

  bool HasInterceptor(int64_t interceptor_id) const;

  int32_t carrier_id() const { return carrier_id_; }

 private:
  DISABLE_COPY_AND_ASSIGN(Carrier);
  Carrier() = delete;

  // create each Interceptor
  void CreateInterceptors();

  int64_t GetRank(int64_t interceptor_id) const;

  // interceptor logic id to actually interceptor
  std::unordered_map<int64_t, std::unique_ptr<Interceptor>>
      interceptor_idx_to_interceptor_;

  Interceptor* source_interceptor_;
  Interceptor* sink_interceptor_;

  bool is_init_{false};

  std::vector<framework::Scope*> microbatch_scopes_;
  framework::Scope* root_scope_{nullptr};
  framework::Scope* minibatch_scope_{nullptr};
  paddle::platform::Place place_;
  paddle::platform::DeviceContext* dev_ctx_{nullptr};
  int64_t rank_;
  int32_t carrier_id_;
  std::unordered_map<int64_t, TaskNode*> interceptor_id_to_node_;
  std::unordered_map<int64_t, int64_t> interceptor_id_to_rank_;
  int thread_num_;
  TaskLoopThreadPool* thread_pool_;
  std::unordered_set<int64_t> interceptor_ids_;

  std::deque<InterceptorMessage> messages_for_test_;
  std::thread test_thread_;
  std::chrono::time_point<std::chrono::steady_clock> cache_begin_;

  void loop_to_send_msg();
};

}  // namespace distributed
}  // namespace paddle
