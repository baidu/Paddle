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

#include "paddle/fluid/distributed/ps/service/graph_brpc_server.h"

#include <string>
#include <thread>  // NOLINT
#include <utility>

#include "butil/endpoint.h"
#include "iomanip"
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_server.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/phi/core/platform/profiler.h"
namespace paddle::distributed {

#define CHECK_TABLE_EXIST(table, request, response)        \
  if (table == NULL) {                                     \
    std::string err_msg("table not found with table_id:"); \
    err_msg.append(std::to_string(request.table_id()));    \
    set_response_code(response, -1, err_msg.c_str());      \
    return -1;                                             \
  }

int32_t GraphBrpcServer::Initialize() {
  auto &service_config = _config.downpour_server_param().service_param();
  if (!service_config.has_service_class()) {
    LOG(ERROR) << "miss service_class in ServerServiceParameter";
    return -1;
  }
  auto *service =
      CREATE_PSCORE_CLASS(PsBaseService, service_config.service_class());
  if (service == nullptr) {
    LOG(ERROR) << "service is unregistered, service_name:"
               << service_config.service_class();
    return -1;
  }

  _service.reset(service);
  if (service->Configure(this) != 0 || service->Initialize() != 0) {
    LOG(ERROR) << "service initialize failed, service_name:"
               << service_config.service_class();
    return -1;
  }
  if (_server.AddService(service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "service add to brpc failed, service:"
               << service_config.service_class();
    return -1;
  }
  return 0;
}

brpc::Channel *GraphBrpcServer::GetCmdChannel(size_t server_index) {
  return _pserver_channels[server_index].get();
}

uint64_t GraphBrpcServer::Start(const std::string &ip, uint32_t port) {
  std::unique_lock<std::mutex> lock(mutex_);

  std::string ip_port = ip + ":" + std::to_string(port);
  VLOG(3) << "server of rank " << _rank << " starts at " << ip_port;
  brpc::ServerOptions options;

  int num_threads = std::thread::hardware_concurrency();
  auto trainers = _environment->GetTrainers();
  options.num_threads = trainers > num_threads ? trainers : num_threads;

  if (_server.Start(ip_port.c_str(), &options) != 0) {
    LOG(ERROR) << "GraphBrpcServer start failed, ip_port=" << ip_port;
    return 0;
  }
  _environment->RegisterPsServer(ip, port, _rank);
  return 0;
}

int32_t GraphBrpcServer::build_peer2peer_connection(int rank) {
  this->rank = rank;
  auto _env = Environment();
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.timeout_ms = 500000;
  options.connection_type = "pooled";
  options.connect_timeout_ms = 10000;
  options.max_retry = 3;

  std::vector<PSHost> server_list = _env->GetPsServers();
  _pserver_channels.resize(server_list.size());
  std::ostringstream os;
  std::string server_ip_port;
  for (size_t i = 0; i < server_list.size(); ++i) {
    server_ip_port.assign(server_list[i].ip.c_str());
    server_ip_port.append(":");
    server_ip_port.append(std::to_string(server_list[i].port));
    _pserver_channels[i].reset(new brpc::Channel());
    if (_pserver_channels[i]->Init(server_ip_port.c_str(), "", &options) != 0) {
      VLOG(0) << "GraphServer connect to Server:" << server_ip_port
              << " Failed! Try again.";
      std::string int_ip_port =
          GetIntTypeEndpoint(server_list[i].ip, server_list[i].port);
      if (_pserver_channels[i]->Init(int_ip_port.c_str(), "", &options) != 0) {
        LOG(ERROR) << "GraphServer connect to Server:" << int_ip_port
                   << " Failed!";
        return -1;
      }
    }
    os << server_ip_port << ",";
  }
  LOG(INFO) << "servers peer2peer connection success:" << os.str();
  return 0;
}

int32_t GraphBrpcService::clear_nodes(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  GraphTableType type_id =
      *(reinterpret_cast<const GraphTableType *>(request.params(0).c_str()));
  int idx_ = *(reinterpret_cast<const int *>(request.params(1).c_str()));
  (reinterpret_cast<GraphTable *>(table))->clear_nodes(type_id, idx_);
  return 0;
}

int32_t GraphBrpcService::add_graph_node(Table *table,
                                         const PsRequestMessage &request,
                                         PsResponseMessage &response,
                                         brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 2) {
    set_response_code(
        response, -1, "add_graph_node request requires at least 2 arguments");
    return 0;
  }

  int idx_ = *reinterpret_cast<const int *>(request.params(0).c_str());
  size_t node_num = request.params(1).size() / sizeof(int64_t);
  const uint64_t *node_data =
      reinterpret_cast<const uint64_t *>(request.params(1).c_str());
  std::vector<uint64_t> node_ids(node_data, node_data + node_num);
  std::vector<bool> is_weighted_list;
  if (request.params_size() == 3) {
    size_t weight_list_size = request.params(2).size() / sizeof(bool);
    const bool *is_weighted_buffer =
        reinterpret_cast<const bool *>(request.params(2).c_str());
    is_weighted_list = std::vector<bool>(is_weighted_buffer,
                                         is_weighted_buffer + weight_list_size);
  }
  // if (request.params_size() == 2) {
  //   size_t weight_list_size = request.params(1).size() / sizeof(bool);
  //   bool *is_weighted_buffer = (bool *)(request.params(1).c_str());
  //   is_weighted_list = std::vector<bool>(is_weighted_buffer,
  //                                        is_weighted_buffer +
  //                                        weight_list_size);
  // }

  (reinterpret_cast<GraphTable *>(table))
      ->add_graph_node(idx_, node_ids, is_weighted_list);
  return 0;
}
int32_t GraphBrpcService::remove_graph_node(Table *table,
                                            const PsRequestMessage &request,
                                            PsResponseMessage &response,
                                            brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 2) {
    set_response_code(
        response,
        -1,
        "remove_graph_node request requires at least 2 arguments");
    return 0;
  }
  int idx_ = *reinterpret_cast<const int *>(request.params(0).c_str());
  size_t node_num = request.params(1).size() / sizeof(uint64_t);
  const uint64_t *node_data =
      reinterpret_cast<const uint64_t *>(request.params(1).c_str());
  std::vector<uint64_t> node_ids(node_data, node_data + node_num);

  (reinterpret_cast<GraphTable *>(table))->remove_graph_node(idx_, node_ids);
  return 0;
}
int32_t GraphBrpcServer::Port() { return _server.listen_address().port; }

int32_t GraphBrpcService::Initialize() {
  _is_initialize_shard_info = false;
  _service_handler_map[PS_STOP_SERVER] = &GraphBrpcService::StopServer;
  _service_handler_map[PS_LOAD_ONE_TABLE] = &GraphBrpcService::LoadOneTable;
  _service_handler_map[PS_LOAD_ALL_TABLE] = &GraphBrpcService::LoadAllTable;

  _service_handler_map[PS_PRINT_TABLE_STAT] = &GraphBrpcService::PrintTableStat;
  _service_handler_map[PS_BARRIER] = &GraphBrpcService::Barrier;
  _service_handler_map[PS_START_PROFILER] = &GraphBrpcService::StartProfiler;
  _service_handler_map[PS_STOP_PROFILER] = &GraphBrpcService::StopProfiler;

  _service_handler_map[PS_PULL_GRAPH_LIST] = &GraphBrpcService::pull_graph_list;
  _service_handler_map[PS_GRAPH_SAMPLE_NEIGHBORS] =
      &GraphBrpcService::graph_random_sample_neighbors;
  _service_handler_map[PS_GRAPH_SAMPLE_NODES] =
      &GraphBrpcService::graph_random_sample_nodes;
  _service_handler_map[PS_GRAPH_GET_NODE_FEAT] =
      &GraphBrpcService::graph_get_node_feat;
  _service_handler_map[PS_GRAPH_CLEAR] = &GraphBrpcService::clear_nodes;
  _service_handler_map[PS_GRAPH_ADD_GRAPH_NODE] =
      &GraphBrpcService::add_graph_node;
  _service_handler_map[PS_GRAPH_REMOVE_GRAPH_NODE] =
      &GraphBrpcService::remove_graph_node;
  _service_handler_map[PS_GRAPH_SET_NODE_FEAT] =
      &GraphBrpcService::graph_set_node_feat;
  _service_handler_map[PS_GRAPH_SAMPLE_NODES_FROM_ONE_SERVER] =
      &GraphBrpcService::sample_neighbors_across_multi_servers;
  InitializeShardInfo();

  return 0;
}

int32_t GraphBrpcService::InitializeShardInfo() {
  if (!_is_initialize_shard_info) {
    std::lock_guard<std::mutex> guard(_initialize_shard_mutex);
    if (_is_initialize_shard_info) {
      return 0;
    }
    server_size = _server->Environment()->GetPsServers().size();
    auto &table_map = *(_server->GetTable());
    for (auto itr : table_map) {
      itr.second->SetShard(_rank, server_size);
    }
    _is_initialize_shard_info = true;
  }
  return 0;
}

void GraphBrpcService::service(google::protobuf::RpcController *cntl_base,
                               const PsRequestMessage *request,
                               PsResponseMessage *response,
                               google::protobuf::Closure *done) {
  brpc::ClosureGuard done_guard(done);
  std::string log_label("ReceiveCmd-");
  if (!request->has_table_id()) {
    set_response_code(*response, -1, "PsRequestMessage.table_id is required");
    return;
  }

  response->set_err_code(0);
  response->set_err_msg("");
  auto *table = _server->GetTable(request->table_id());
  brpc::Controller *cntl = static_cast<brpc::Controller *>(cntl_base);
  auto itr = _service_handler_map.find(request->cmd_id());
  if (itr == _service_handler_map.end()) {
    std::string err_msg(
        "undefined cmd_id, should match PsCmdID in ps.proto, cmd_id:");
    err_msg.append(std::to_string(request->cmd_id()));
    set_response_code(*response, -1, err_msg.c_str());
    return;
  }
  serviceFunc handler_func = itr->second;
  int service_ret = (this->*handler_func)(table, *request, *response, cntl);
  if (service_ret != 0) {
    response->set_err_code(service_ret);
    if (!response->has_err_msg()) {
      response->set_err_msg("server internal error");
    }
  }
}

int32_t GraphBrpcService::Barrier(Table *table,
                                  const PsRequestMessage &request,
                                  PsResponseMessage &response,
                                  brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)

  if (request.params_size() < 1) {
    set_response_code(response,
                      -1,
                      "PsRequestMessage.params is required at "
                      "least 1 for num of sparse_key");
    return 0;
  }

  auto trainer_id = request.client_id();
  auto barrier_type = request.params(0);
  table->Barrier(trainer_id, barrier_type);
  return 0;
}

int32_t GraphBrpcService::PrintTableStat(Table *table,
                                         const PsRequestMessage &request,
                                         PsResponseMessage &response,
                                         brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  std::pair<int64_t, int64_t> ret = table->PrintTableStat();
  paddle::framework::BinaryArchive ar;
  ar << ret.first << ret.second;
  std::string table_info(ar.Buffer(), ar.Length());
  response.set_data(table_info);

  return 0;
}

int32_t GraphBrpcService::LoadOneTable(Table *table,
                                       const PsRequestMessage &request,
                                       PsResponseMessage &response,
                                       brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 2) {
    set_response_code(
        response,
        -1,
        "PsRequestMessage.datas is required at least 2 for path & load_param");
    return -1;
  }
  if (table->Load(request.params(0), request.params(1)) != 0) {
    set_response_code(response, -1, "table load failed");
    return -1;
  }
  return 0;
}

int32_t GraphBrpcService::LoadAllTable(Table *table,
                                       const PsRequestMessage &request,
                                       PsResponseMessage &response,
                                       brpc::Controller *cntl) {
  auto &table_map = *(_server->GetTable());
  for (auto &itr : table_map) {
    if (LoadOneTable(itr.second.get(), request, response, cntl) != 0) {
      LOG(ERROR) << "load table[" << itr.first << "] failed";
      return -1;
    }
  }
  return 0;
}

int32_t GraphBrpcService::StopServer(Table *table,
                                     const PsRequestMessage &request,
                                     PsResponseMessage &response,
                                     brpc::Controller *cntl) {
  GraphBrpcServer *p_server = reinterpret_cast<GraphBrpcServer *>(_server);
  std::thread t_stop([p_server]() {
    p_server->Stop();
    LOG(INFO) << "Server Stopped";
  });
  p_server->export_cv()->notify_all();
  t_stop.detach();
  return 0;
}

int32_t GraphBrpcService::StopProfiler(Table *table,
                                       const PsRequestMessage &request,
                                       PsResponseMessage &response,
                                       brpc::Controller *cntl) {
  platform::DisableProfiler(platform::EventSortingKey::kDefault,
                            string::Sprintf("server_%s_profile", _rank));
  return 0;
}

int32_t GraphBrpcService::StartProfiler(Table *table,
                                        const PsRequestMessage &request,
                                        PsResponseMessage &response,
                                        brpc::Controller *cntl) {
  platform::EnableProfiler(platform::ProfilerState::kCPU);
  return 0;
}

int32_t GraphBrpcService::pull_graph_list(Table *table,
                                          const PsRequestMessage &request,
                                          PsResponseMessage &response,
                                          brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 5) {
    set_response_code(
        response, -1, "pull_graph_list request requires at least 5 arguments");
    return 0;
  }
  GraphTableType type_id =
      *(reinterpret_cast<const GraphTableType *>(request.params(0).c_str()));
  int idx = *(reinterpret_cast<const int *>(request.params(1).c_str()));
  int start = *(reinterpret_cast<const int *>(request.params(2).c_str()));
  int size = *(reinterpret_cast<const int *>(request.params(3).c_str()));
  int step = *(reinterpret_cast<const int *>(request.params(4).c_str()));
  std::unique_ptr<char[]> buffer;
  int actual_size;
  (reinterpret_cast<GraphTable *>(table))
      ->pull_graph_list(
          type_id, idx, start, size, buffer, actual_size, false, step);
  cntl->response_attachment().append(buffer.get(), actual_size);
  return 0;
}
int32_t GraphBrpcService::graph_random_sample_neighbors(
    Table *table,
    const PsRequestMessage &request,
    PsResponseMessage &response,
    brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 4) {
    set_response_code(
        response,
        -1,
        "graph_random_sample_neighbors request requires at least 3 arguments");
    return 0;
  }
  int idx_ = *reinterpret_cast<const int *>(request.params(0).c_str());
  size_t node_num = request.params(1).size() / sizeof(uint64_t);
  uint64_t *node_data = (uint64_t *)(request.params(1).c_str());  // NOLINT
  const int sample_size =
      *reinterpret_cast<const int *>(request.params(2).c_str());
  const bool need_weight =
      *reinterpret_cast<const bool *>(request.params(3).c_str());
  std::vector<std::shared_ptr<char>> buffers(node_num);
  std::vector<int> actual_sizes(node_num, 0);
  (reinterpret_cast<GraphTable *>(table))
      ->random_sample_neighbors(
          idx_, node_data, sample_size, buffers, actual_sizes, need_weight);

  cntl->response_attachment().append(&node_num, sizeof(size_t));
  cntl->response_attachment().append(actual_sizes.data(),
                                     sizeof(int) * node_num);
  for (size_t idx = 0; idx < node_num; ++idx) {
    cntl->response_attachment().append(buffers[idx].get(), actual_sizes[idx]);
  }
  return 0;
}
int32_t GraphBrpcService::graph_random_sample_nodes(
    Table *table,
    const PsRequestMessage &request,
    PsResponseMessage &response,
    brpc::Controller *cntl) {
  GraphTableType type_id =
      *(reinterpret_cast<const GraphTableType *>(request.params(0).c_str()));
  int idx_ = *(reinterpret_cast<const int *>(request.params(1).c_str()));
  size_t size =
      *(reinterpret_cast<const uint64_t *>(request.params(2).c_str()));
  // size_t size = *(int64_t *)(request.params(0).c_str());
  std::unique_ptr<char[]> buffer;
  int actual_size;
  if (reinterpret_cast<GraphTable *>(table)->random_sample_nodes(
          type_id, idx_, size, buffer, actual_size) == 0) {
    cntl->response_attachment().append(buffer.get(), actual_size);
  } else {
    cntl->response_attachment().append(nullptr, 0);
  }

  return 0;
}

int32_t GraphBrpcService::graph_get_node_feat(Table *table,
                                              const PsRequestMessage &request,
                                              PsResponseMessage &response,
                                              brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 3) {
    set_response_code(
        response,
        -1,
        "graph_get_node_feat request requires at least 3 arguments");
    return 0;
  }
  int idx_ = *reinterpret_cast<const int *>(request.params(0).c_str());
  size_t node_num = request.params(1).size() / sizeof(uint64_t);
  const uint64_t *node_data =
      reinterpret_cast<const uint64_t *>(request.params(1).c_str());
  std::vector<uint64_t> node_ids(node_data, node_data + node_num);

  std::vector<std::string> feature_names =
      paddle::string::split_string<std::string>(request.params(2), "\t");

  std::vector<std::vector<std::string>> feature(
      feature_names.size(), std::vector<std::string>(node_num));

  (reinterpret_cast<GraphTable *>(table))
      ->get_node_feat(idx_, node_ids, feature_names, feature);

  for (size_t feat_idx = 0; feat_idx < feature_names.size(); ++feat_idx) {
    for (size_t node_idx = 0; node_idx < node_num; ++node_idx) {
      size_t feat_len = feature[feat_idx][node_idx].size();
      cntl->response_attachment().append(&feat_len, sizeof(size_t));
      cntl->response_attachment().append(feature[feat_idx][node_idx].data(),
                                         feat_len);
    }
  }

  return 0;
}
int32_t GraphBrpcService::sample_neighbors_across_multi_servers(
    Table *table,
    const PsRequestMessage &request,
    PsResponseMessage &response,
    brpc::Controller *cntl) {
  // sleep(5);
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 4) {
    set_response_code(response,
                      -1,
                      "sample_neighbors_across_multi_servers request requires "
                      "at least 4 arguments");
    return 0;
  }

  int idx_ = *reinterpret_cast<const int *>(request.params(0).c_str());
  size_t node_num = request.params(1).size() / sizeof(uint64_t);
  const uint64_t *node_data =
      reinterpret_cast<const uint64_t *>(request.params(1).c_str());
  int sample_size = *reinterpret_cast<const int *>(request.params(2).c_str());
  bool need_weight = *reinterpret_cast<const bool *>(request.params(3).c_str());

  std::vector<int> request2server;
  std::vector<int> server2request(server_size, -1);
  std::vector<uint64_t> local_id;
  std::vector<int> local_query_idx;
  size_t rank = GetRank();
  for (size_t query_idx = 0; query_idx < node_num; ++query_idx) {
    int server_index = (reinterpret_cast<GraphTable *>(table))
                           ->get_server_index_by_id(node_data[query_idx]);
    if (server2request[server_index] == -1) {
      server2request[server_index] = request2server.size();
      request2server.push_back(server_index);
    }
  }
  if (server2request[rank] != -1) {
    auto pos = server2request[rank];
    std::swap(request2server[pos],
              request2server[static_cast<int>(request2server.size()) - 1]);
    server2request[request2server[pos]] = pos;
    server2request[request2server[static_cast<int>(request2server.size()) -
                                  1]] = request2server.size() - 1;
  }
  size_t request_call_num = request2server.size();
  std::vector<std::shared_ptr<char>> local_buffers;
  std::vector<int> local_actual_sizes;
  std::vector<size_t> seq;
  std::vector<std::vector<uint64_t>> node_id_buckets(request_call_num);
  std::vector<std::vector<int>> query_idx_buckets(request_call_num);
  for (size_t query_idx = 0; query_idx < node_num; ++query_idx) {
    int server_index = (reinterpret_cast<GraphTable *>(table))
                           ->get_server_index_by_id(node_data[query_idx]);
    int request_idx = server2request[server_index];
    node_id_buckets[request_idx].push_back(node_data[query_idx]);
    query_idx_buckets[request_idx].push_back(query_idx);
    seq.push_back(request_idx);
  }
  size_t remote_call_num = request_call_num;
  if (!request2server.empty() &&
      static_cast<size_t>(request2server.back()) == rank) {
    remote_call_num--;
    local_buffers.resize(node_id_buckets.back().size());
    local_actual_sizes.resize(node_id_buckets.back().size());
  }
  cntl->response_attachment().append(&node_num, sizeof(size_t));
  auto local_promise = std::make_shared<std::promise<int32_t>>();
  std::future<int> local_fut = local_promise->get_future();
  std::vector<bool> failed(server_size, false);
  std::function<void(void *)> func = [&, node_id_buckets, query_idx_buckets](
                                         void *done) {
    local_fut.get();
    std::vector<int> actual_size;
    auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
    std::vector<std::unique_ptr<butil::IOBufBytesIterator>> res(
        remote_call_num);
    size_t fail_num = 0;
    for (size_t request_idx = 0; request_idx < remote_call_num; ++request_idx) {
      if (closure->check_response(request_idx, PS_GRAPH_SAMPLE_NEIGHBORS) !=
          0) {
        ++fail_num;
        failed[request2server[request_idx]] = true;
      } else {
        auto &res_io_buffer = closure->cntl(request_idx)->response_attachment();
        res[request_idx].reset(new butil::IOBufBytesIterator(res_io_buffer));
        size_t num;
        res[request_idx]->copy_and_forward(&num, sizeof(size_t));
      }
    }
    int size;
    int local_index = 0;
    for (size_t i = 0; i < node_num; i++) {
      if (fail_num > 0 && failed[seq[i]]) {
        size = 0;
      } else if (static_cast<size_t>(request2server[seq[i]]) != rank) {
        res[seq[i]]->copy_and_forward(&size, sizeof(int));
      } else {
        size = local_actual_sizes[local_index++];
      }
      actual_size.push_back(size);
    }
    cntl->response_attachment().append(actual_size.data(),
                                       actual_size.size() * sizeof(int));

    local_index = 0;
    for (size_t i = 0; i < node_num; i++) {
      if (fail_num > 0 && failed[seq[i]]) {
        continue;
      } else if (static_cast<size_t>(request2server[seq[i]]) != rank) {
        std::vector<char> temp(actual_size[i] + 1);
        res[seq[i]]->copy_and_forward(temp.data(), actual_size[i]);
        cntl->response_attachment().append(temp.data(), actual_size[i]);
      } else {
        char *temp = local_buffers[local_index++].get();
        cntl->response_attachment().append(temp, actual_size[i]);
      }
    }
    closure->set_promise_value(0);
  };

  DownpourBrpcClosure *closure = new DownpourBrpcClosure(remote_call_num, func);

  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  for (size_t request_idx = 0; request_idx < remote_call_num; ++request_idx) {
    int server_index = request2server[request_idx];
    closure->request(request_idx)->set_cmd_id(PS_GRAPH_SAMPLE_NEIGHBORS);
    closure->request(request_idx)->set_table_id(request.table_id());
    closure->request(request_idx)->set_client_id(rank);
    size_t node_num = node_id_buckets[request_idx].size();

    closure->request(request_idx)
        ->add_params(reinterpret_cast<char *>(&idx_), sizeof(int));

    closure->request(request_idx)
        ->add_params(
            reinterpret_cast<char *>(node_id_buckets[request_idx].data()),
            sizeof(uint64_t) * node_num);
    closure->request(request_idx)
        ->add_params(reinterpret_cast<char *>(&sample_size), sizeof(int));
    closure->request(request_idx)
        ->add_params(reinterpret_cast<char *>(&need_weight), sizeof(bool));
    PsService_Stub rpc_stub((reinterpret_cast<GraphBrpcServer *>(GetServer())
                                 ->GetCmdChannel(server_index)));
    // GraphPsService_Stub rpc_stub =
    //     getServiceStub(GetCmdChannel(server_index));
    closure->cntl(request_idx)->set_log_id(butil::gettimeofday_ms());
    rpc_stub.service(closure->cntl(request_idx),
                     closure->request(request_idx),
                     closure->response(request_idx),
                     closure);
  }
  if (server2request[rank] != -1) {
    (reinterpret_cast<GraphTable *>(table))
        ->random_sample_neighbors(idx_,
                                  node_id_buckets.back().data(),
                                  sample_size,
                                  local_buffers,
                                  local_actual_sizes,
                                  need_weight);
  }
  local_promise.get()->set_value(0);
  if (remote_call_num == 0) func(closure);
  fut.get();
  return 0;
}
int32_t GraphBrpcService::graph_set_node_feat(Table *table,
                                              const PsRequestMessage &request,
                                              PsResponseMessage &response,
                                              brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 4) {
    set_response_code(
        response,
        -1,
        "graph_set_node_feat request requires at least 3 arguments");
    return 0;
  }
  int idx_ = *reinterpret_cast<const int *>(request.params(0).c_str());

  size_t node_num = request.params(1).size() / sizeof(uint64_t);
  const uint64_t *node_data =
      reinterpret_cast<const uint64_t *>(request.params(1).c_str());
  std::vector<uint64_t> node_ids(node_data, node_data + node_num);

  // std::vector<std::string> feature_names =
  //     paddle::string::split_string<std::string>(request.params(1), "\t");

  std::vector<std::string> feature_names =
      paddle::string::split_string<std::string>(request.params(2), "\t");

  std::vector<std::vector<std::string>> features(
      feature_names.size(), std::vector<std::string>(node_num));

  //  const char *buffer = request.params(2).c_str();
  const char *buffer = request.params(3).c_str();

  for (size_t feat_idx = 0; feat_idx < feature_names.size(); ++feat_idx) {
    for (size_t node_idx = 0; node_idx < node_num; ++node_idx) {
      const size_t feat_len = *reinterpret_cast<const size_t *>(buffer);
      buffer += sizeof(size_t);
      auto feat = std::string(buffer, feat_len);
      features[feat_idx][node_idx] = feat;
      buffer += feat_len;
    }
  }

  (reinterpret_cast<GraphTable *>(table))
      ->set_node_feat(idx_, node_ids, feature_names, features);

  return 0;
}

}  // namespace paddle::distributed
