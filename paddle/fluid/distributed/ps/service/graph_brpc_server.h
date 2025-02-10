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

#include <memory>
#include <vector>

#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_server.h"
#include "paddle/fluid/distributed/ps/service/server.h"
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/distributed/ps/table/table.h"
namespace paddle {
namespace distributed {
class GraphBrpcServer : public PSServer {
 public:
  GraphBrpcServer() {}
  virtual ~GraphBrpcServer() {}
  PsBaseService *get_service() { return _service.get(); }
  virtual uint64_t Start(const std::string &ip, uint32_t port);
  virtual int32_t build_peer2peer_connection(int rank);
  virtual brpc::Channel *GetCmdChannel(size_t server_index);
  virtual int32_t Stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_) return 0;
    stopped_ = true;
    // cv_.notify_all();
    _server.Stop(1000);
    _server.Join();
    return 0;
  }
  int32_t Port();

  std::condition_variable *export_cv() { return &cv_; }

 private:
  virtual int32_t Initialize();
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool stopped_ = false;
  int rank;
  brpc::Server _server;
  std::shared_ptr<PsBaseService> _service;
  std::vector<std::shared_ptr<brpc::Channel>> _pserver_channels;
};

class GraphBrpcService;

typedef int32_t (GraphBrpcService::*serviceFunc)(
    Table *table,
    const PsRequestMessage &request,
    PsResponseMessage &response,  // NOLINT
    brpc::Controller *cntl);

class GraphBrpcService : public PsBaseService {
 public:
  int32_t Initialize() override;

  void service(::google::protobuf::RpcController *controller,
               const PsRequestMessage *request,
               PsResponseMessage *response,
               ::google::protobuf::Closure *done) override;

 protected:
  std::unordered_map<int32_t, serviceFunc> _service_handler_map;
  int32_t InitializeShardInfo();
  int32_t pull_graph_list(Table *table,
                          const PsRequestMessage &request,
                          PsResponseMessage &response,  // NOLINT
                          brpc::Controller *cntl);
  int32_t graph_random_sample_neighbors(Table *table,
                                        const PsRequestMessage &request,
                                        PsResponseMessage &response,  // NOLINT
                                        brpc::Controller *cntl);
  int32_t graph_random_sample_nodes(Table *table,
                                    const PsRequestMessage &request,
                                    PsResponseMessage &response,  // NOLINT
                                    brpc::Controller *cntl);

  int32_t graph_get_node_feat(Table *table,
                              const PsRequestMessage &request,
                              PsResponseMessage &response,  // NOLINT
                              brpc::Controller *cntl);
  int32_t graph_set_node_feat(Table *table,
                              const PsRequestMessage &request,
                              PsResponseMessage &response,  // NOLINT
                              brpc::Controller *cntl);
  int32_t clear_nodes(Table *table,
                      const PsRequestMessage &request,
                      PsResponseMessage &response,  // NOLINT
                      brpc::Controller *cntl);
  int32_t add_graph_node(Table *table,
                         const PsRequestMessage &request,
                         PsResponseMessage &response,  // NOLINT
                         brpc::Controller *cntl);
  int32_t remove_graph_node(Table *table,
                            const PsRequestMessage &request,
                            PsResponseMessage &response,  // NOLINT
                            brpc::Controller *cntl);
  int32_t Barrier(Table *table,
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

  int32_t sample_neighbors_across_multi_servers(
      Table *table,
      const PsRequestMessage &request,
      PsResponseMessage &response,  // NOLINT
      brpc::Controller *cntl);

  int32_t use_neighbors_sample_cache(Table *table,
                                     const PsRequestMessage &request,
                                     PsResponseMessage &response,  // NOLINT
                                     brpc::Controller *cntl);

  int32_t load_graph_split_config(Table *table,
                                  const PsRequestMessage &request,
                                  PsResponseMessage &response,  // NOLINT
                                  brpc::Controller *cntl);

 private:
  bool _is_initialize_shard_info;
  std::mutex _initialize_shard_mutex;
  std::unordered_map<int32_t, serviceHandlerFunc> _msg_handler_map;
  std::vector<float> _ori_values;
  const int sample_nodes_ranges = 23;
  size_t server_size;
  std::shared_ptr<::ThreadPool> task_pool;
};

}  // namespace distributed
}  // namespace paddle
