// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/ps/service/brpc_utils.h"
#include "paddle/fluid/distributed/ps/service/server.h"

namespace brpc {
class Controller;
}  // namespace brpc
namespace google {
namespace protobuf {
class Closure;
class RpcController;
}  // namespace protobuf
}  // namespace google

namespace paddle {
namespace distributed {

class PsRequestMessage;
class PsResponseMessage;
class Table;

class BrpcPsServer : public PSServer {
 public:
  BrpcPsServer() {}
  virtual ~BrpcPsServer() {}
  virtual uint64_t Start(const std::string &ip, uint32_t port);
  virtual int32_t Stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    stopped_ = true;
    cv_.notify_all();

    _server.Stop(1000);
    _server.Join();
    return 0;
  }
  int32_t Port();

  int32_t StartS2S() override;
  ::std::future<int32_t> SendPServer2PServerMsg(
      int msg_type, int to_pserver_id, const std::string &msg) override;
  int32_t ReceiveFromPServer(int msg_type,
                             int pserver_id,
                             const std::string &msg) override;

 private:
  virtual int32_t Initialize();
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool stopped_ = false;
  brpc::Server _server;
  std::shared_ptr<PsBaseService> _service;
  std::vector<std::shared_ptr<brpc::Channel>> _pserver_channels;
};

class BrpcPsService;

typedef int32_t (BrpcPsService::*serviceHandlerFunc)(
    Table *table,
    const PsRequestMessage &request,
    PsResponseMessage &response,  // NOLINT
    brpc::Controller *cntl);

class BrpcPsService : public PsBaseService {
 public:
  int32_t Initialize() override;

  void service(::google::protobuf::RpcController *controller,
               const PsRequestMessage *request,
               PsResponseMessage *response,
               ::google::protobuf::Closure *done) override;

 private:
  int32_t InitializeShardInfo();
  int32_t PullDense(Table *table,
                    const PsRequestMessage &request,
                    PsResponseMessage &response,  // NOLINT
                    brpc::Controller *cntl);
  int32_t PushDense(Table *table,
                    const PsRequestMessage &request,
                    PsResponseMessage &response,  // NOLINT
                    brpc::Controller *cntl);
  int32_t PushDenseParam(Table *table,
                         const PsRequestMessage &request,
                         PsResponseMessage &response,  // NOLINT
                         brpc::Controller *cntl);
  int32_t PushSparseParam(Table *table,
                          const PsRequestMessage &request,
                          PsResponseMessage &response,  // NOLINT
                          brpc::Controller *cntl);
  int32_t PullSparse(Table *table,
                     const PsRequestMessage &request,
                     PsResponseMessage &response,  // NOLINT
                     brpc::Controller *cntl);
  int32_t PullGeoParam(Table *table,
                       const PsRequestMessage &request,
                       PsResponseMessage &response,  // NOLINT
                       brpc::Controller *cntl);
  int32_t Barrier(Table *table,
                  const PsRequestMessage &request,
                  PsResponseMessage &response,  // NOLINT
                  brpc::Controller *cntl);
  int32_t PushSparse(Table *table,
                     const PsRequestMessage &request,
                     PsResponseMessage &response,  // NOLINT
                     brpc::Controller *cntl);
  int32_t LoadOneTable(Table *table,
                       const PsRequestMessage &request,
                       PsResponseMessage &response,  // NOLINT
                       brpc::Controller *cntl);
  int32_t LoadAllTable(Table *table,
                       const PsRequestMessage &request,
                       PsResponseMessage &response,  // NOLINT
                       brpc::Controller *cntl);
  int32_t SaveOneTable(Table *table,
                       const PsRequestMessage &request,
                       PsResponseMessage &response,  // NOLINT
                       brpc::Controller *cntl);
  int32_t SaveAllTable(Table *table,
                       const PsRequestMessage &request,
                       PsResponseMessage &response,  // NOLINT
                       brpc::Controller *cntl);
  int32_t ShrinkTable(Table *table,
                      const PsRequestMessage &request,
                      PsResponseMessage &response,  // NOLINT
                      brpc::Controller *cntl);
  int32_t ClearOneTable(Table *table,
                        const PsRequestMessage &request,
                        PsResponseMessage &response,  // NOLINT
                        brpc::Controller *cntl);
  int32_t ClearAllTable(Table *table,
                        const PsRequestMessage &request,
                        PsResponseMessage &response,  // NOLINT
                        brpc::Controller *cntl);
  int32_t StopServer(Table *table,
                     const PsRequestMessage &request,
                     PsResponseMessage &response,  // NOLINT
                     brpc::Controller *cntl);
  int32_t StartProfiler(Table *table,
                        const PsRequestMessage &request,
                        PsResponseMessage &response,  // NOLINT
                        brpc::Controller *cntl);
  int32_t StopProfiler(Table *table,
                       const PsRequestMessage &request,
                       PsResponseMessage &response,  // NOLINT
                       brpc::Controller *cntl);

  int32_t PrintTableStat(Table *table,
                         const PsRequestMessage &request,
                         PsResponseMessage &response,  // NOLINT
                         brpc::Controller *cntl);

  int32_t PushGlobalStep(Table *table,
                         const PsRequestMessage &request,
                         PsResponseMessage &response,  // NOLINT
                         brpc::Controller *cntl);

  int32_t CacheShuffle(Table *table,
                       const PsRequestMessage &request,
                       PsResponseMessage &response,  // NOLINT
                       brpc::Controller *cntl);

  int32_t SaveCacheTable(Table *table,
                         const PsRequestMessage &request,
                         PsResponseMessage &response,  // NOLINT
                         brpc::Controller *cntl);

  int32_t GetCacheThreshold(Table *table,
                            const PsRequestMessage &request,
                            PsResponseMessage &response,  // NOLINT
                            brpc::Controller *cntl);

  int32_t Revert(Table *table,
                 const PsRequestMessage &request,
                 PsResponseMessage &response,  // NOLINT
                 brpc::Controller *cntl);

  int32_t CheckSavePrePatchDone(Table *table,
                                const PsRequestMessage &request,
                                PsResponseMessage &response,  // NOLINT
                                brpc::Controller *cntl);

  bool _is_initialize_shard_info;
  std::mutex _initialize_shard_mutex;
  std::unordered_map<int32_t, serviceHandlerFunc> _service_handler_map;
  std::unordered_map<int32_t, serviceHandlerFunc> _msg_handler_map;
  std::vector<float> _ori_values;
};

class DownpourPServerBrpcClosure : public PServerClosure {
 public:
  DownpourPServerBrpcClosure(size_t num, PServerCallBack callback)
      : PServerClosure(callback) {
    _waiting_num = num;
    _cntls.resize(num);
    _requests.resize(num);
    _responses.resize(num);
    for (size_t i = 0; i < num; ++i) {
      _cntls[i].reset(new brpc::Controller());
    }
  }
  virtual ~DownpourPServerBrpcClosure() {}

  void Run() override {
    if (_waiting_num.fetch_sub(1) == 1) {
      _callback(this);
      delete this;
    }
  }
  PsRequestMessage *request(size_t i) { return &_requests[i]; }
  PsResponseMessage *response(size_t i) { return &_responses[i]; }
  brpc::Controller *cntl(size_t i) { return _cntls[i].get(); }
  int check_response(size_t request_idx UNUSED, int cmd_id UNUSED) { return 1; }
  int check_save_response(size_t request_idx UNUSED, int cmd_id UNUSED) {
    return 1;
  }

 private:
  std::atomic<int32_t> _waiting_num;
  std::vector<PsRequestMessage> _requests;
  std::vector<PsResponseMessage> _responses;
  std::vector<std::shared_ptr<brpc::Controller>> _cntls;
};
}  // namespace distributed
}  // namespace paddle
