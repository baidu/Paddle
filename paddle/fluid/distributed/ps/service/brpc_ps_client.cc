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

#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"

#include <memory>
#include <sstream>
#include <string>

#include "paddle/fluid/distributed/ps/service/coordinator_client.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/utils/string/split.h"

static const int max_port = 65535;

namespace paddle::framework {
class Scope;
class Variable;
}  // namespace paddle::framework

namespace paddle::distributed {

PD_DEFINE_int32(pserver_push_dense_merge_limit,
                12,
                "limit max push_dense local merge requests");

PD_DEFINE_int32(pserver_push_sparse_merge_limit,
                12,
                "limit max push_sparse local merge requests");

PD_DEFINE_int32(pserver_pull_dense_limit,
                12,
                "limit max push_sparse local merge requests");

PD_DEFINE_int32(pserver_async_push_dense_interval_ms,
                10,
                "async push_dense to server interval");

PD_DEFINE_int32(pserver_async_push_sparse_interval_ms,
                10,
                "async push_sparse to server interval");

PD_DEFINE_bool(pserver_scale_gradient_by_merge,
               false,
               "scale dense gradient when merged");

PD_DEFINE_int32(pserver_communicate_compress_type,
                0,
                "none:0 snappy:1 gzip:2 zlib:3 lz4:4");

PD_DEFINE_int32(pserver_max_async_call_num,
                13,
                "max task num in async_call_server");

PD_DEFINE_int32(pserver_timeout_ms,
                500000,
                "pserver request server timeout_ms");

PD_DEFINE_int32(pserver_connect_timeout_ms,
                10000,
                "pserver connect server timeout_ms");

PD_DEFINE_int32(pserver_sparse_merge_thread,
                1,
                "pserver sparse merge thread num");

PD_DEFINE_int32(pserver_sparse_table_shard_num,
                1000,
                "sparse table shard for save & load");

inline size_t get_sparse_shard(uint32_t shard_num,
                               uint32_t server_num,
                               uint64_t key) {
  size_t remind = shard_num % server_num;
  size_t local_shard_num =
      remind == 0 ? shard_num / server_num : shard_num / server_num + 1;
  return (key % shard_num) / local_shard_num;
}

void DownpourPsClientService::service(
    ::google::protobuf::RpcController *controller,
    const PsRequestMessage *request,
    PsResponseMessage *response,
    ::google::protobuf::Closure *done) {
  brpc::ClosureGuard done_guard(done);
  int ret = _client->HandleClient2ClientMsg(
      request->cmd_id(), request->client_id(), request->data());
  response->set_err_code(0);
  response->set_err_msg("");
  if (ret != 0) {
    response->set_err_code(-1);
    response->set_err_msg("handle_client2client_msg failed");
  }
}

// 启动 client 端 RpcService 用于数据互发等操作
int32_t BrpcPsClient::StartClientService() {
  if (_service.Configure(this, _client_id) != 0) {
    LOG(ERROR)
        << "service initialize failed, service_name:DownpourPsClientService";
    return -1;
  }
  _server.AddService(&_service, brpc::SERVER_DOESNT_OWN_SERVICE);
  brpc::ServerOptions options;
  int start_port = 8500;
  options.num_threads = 24;

  if (_server.Start(butil::my_ip_cstr(),
                    brpc::PortRange(start_port, max_port),
                    &options) != 0) {
    LOG(ERROR) << "BrpcPsServer start failed";
    return -1;
  }
  _server_started = true;
  _env->RegisterPsClient(
      butil::my_ip_cstr(), _server.listen_address().port, _client_id);
  VLOG(0) << "BrpcPsClient Service addr: " << butil::my_ip_cstr() << ", "
          << _server.listen_address().port << ", " << _client_id;
  return 0;
}

// 启动 FlClientService，用户接收 coordinator 数据
int32_t BrpcPsClient::StartFlClientService(const std::string &self_endpoint) {
  _fl_server.AddService(&_service, brpc::SERVER_DOESNT_OWN_SERVICE);
  brpc::ServerOptions options;
  if (self_endpoint.empty()) {
    LOG(ERROR) << "fl-ps > fl client endpoint not set";
    return -1;
  }

  if (_fl_server.Start(self_endpoint.c_str(), &options) != 0) {
    VLOG(0) << "fl-ps > StartFlClientService failed. Try again.";
    auto ip_port = ::paddle::string::Split(self_endpoint, ':');
    std::string ip = ip_port[0];
    int port = std::stoi(ip_port[1]);
    std::string int_ip_port = GetIntTypeEndpoint(ip, port);
    if (_fl_server.Start(int_ip_port.c_str(), &options) != 0) {
      LOG(ERROR) << "fl-ps > StartFlClientService failed, ip_port= "
                 << int_ip_port;
      return -1;
    }
  } else {
    VLOG(0) << "fl-ps > StartFlClientService succeed! listen on "
            << self_endpoint;
  }
  return 0;
}

int32_t BrpcPsClient::CreateClient2ClientConnection(
    int pserver_timeout_ms, int pserver_connect_timeout_ms, int max_retry) {
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.timeout_ms = pserver_timeout_ms;
  options.connection_type = "pooled";
  options.connect_timeout_ms = pserver_connect_timeout_ms;
  options.max_retry = max_retry;

  std::vector<PSHost> client_list = _env->GetPsClients();
  VLOG(1) << "BrpcPsClient::create_c2c_connection client_list size: "
          << client_list.size();
  for (auto cc : client_list) {
    VLOG(1) << "BrpcPsClient::create_c2c_connection client_list: "
            << cc.ToString();
  }
  _client_channels.resize(client_list.size());
  std::ostringstream os;
  std::string server_ip_port;
  for (size_t i = 0; i < client_list.size(); ++i) {
    server_ip_port.assign(client_list[i].ip.c_str());
    server_ip_port.append(":");
    server_ip_port.append(std::to_string(client_list[i].port));
    _client_channels[i].reset(new brpc::Channel());
    if (_client_channels[i]->Init(server_ip_port.c_str(), "", &options)) {
      VLOG(0) << "BrpcPSClient connect to Client:" << server_ip_port
              << " Failed! Try again.";
      std::string int_ip_port =
          GetIntTypeEndpoint(client_list[i].ip, client_list[i].port);
      if (_client_channels[i]->Init(int_ip_port.c_str(), "", &options) != 0) {
        LOG(ERROR) << "BrpcPSClient connect to Client:" << int_ip_port
                   << " Failed!";
        return -1;
      }
    }
    os << server_ip_port << ",";
  }
  LOG(INFO) << "Client connect success:" << os.str();
  return 0;
}

int32_t BrpcPsClient::InitializeFlWorker(const std::string &self_endpoint) {
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.timeout_ms = FLAGS_pserver_timeout_ms;
  options.connection_type = "pooled";
  options.connect_timeout_ms = FLAGS_pserver_connect_timeout_ms;
  options.max_retry = 3;
  // 获取 coordinator 列表，并连接
  std::string coordinator_ip_port;
  std::vector<PSHost> coordinator_list = _env->GetCoordinators();
  _coordinator_channels.resize(coordinator_list.size());
  for (size_t i = 0; i < coordinator_list.size(); ++i) {
    coordinator_ip_port.assign(coordinator_list[i].ip.c_str());
    coordinator_ip_port.append(":");
    coordinator_ip_port.append(std::to_string(coordinator_list[i].port));
    VLOG(0) << "fl-ps > BrpcFlclient connecting to coordinator: "
            << coordinator_ip_port;
    for (size_t j = 0; j < _coordinator_channels[i].size(); ++j) {
      _coordinator_channels[i][j].reset(new brpc::Channel());
      if (_coordinator_channels[i][j]->Init(
              coordinator_ip_port.c_str(), "", &options) != 0) {
        LOG(ERROR) << "fl-ps > BrpcFlclient connect to coordinator:"
                   << coordinator_ip_port << " Failed! Try again.";
        std::string int_ip_port = GetIntTypeEndpoint(coordinator_list[i].ip,
                                                     coordinator_list[i].port);
        if (_coordinator_channels[i][j]->Init(
                int_ip_port.c_str(), "", &options) != 0) {
          LOG(ERROR) << "fl-ps > BrpcFlclient connect to coordinator:"
                     << int_ip_port << " Failed!";
          return -1;
        }
      }
    }
  }
  StartFlClientService(self_endpoint);
  VLOG(0) << "fl-ps > InitializeFlWorker finished!";
  return 0;
}

void BrpcPsClient::PushFLClientInfoSync(const std::string &fl_client_info) {
  size_t request_call_num = _coordinator_channels.size();
  FlClientBrpcClosure *closure =
      new FlClientBrpcClosure(request_call_num, [request_call_num](void *done) {
        auto *closure = reinterpret_cast<FlClientBrpcClosure *>(done);
        int ret = 0;
        for (size_t i = 0; i < request_call_num; i++) {
          if (closure->check_response(i, PUSH_FL_CLIENT_INFO_SYNC) != 0) {
            LOG(ERROR) << "fl-ps > PushFLClientInfoSync response from "
                          "coordinator is failed";
            ret = -1;
            return;
          } else {
            VLOG(0) << "fl-ps > rpc service call cost time: "
                    << (closure->cntl(i)->latency_us() / 1000) << " ms";
          }
        }
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  std::future<int32_t> fut = promise->get_future();
  closure->add_promise(promise);
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PUSH_FL_CLIENT_INFO_SYNC);
    closure->request(i)->set_client_id(_client_id);
    closure->request(i)->set_str_params(fl_client_info);
    brpc::Channel *rpc_channel = _coordinator_channels[0][0].get();
    if (rpc_channel == nullptr) {
      LOG(ERROR) << "_coordinator_channels is null";
      return;
    }
    PsService_Stub rpc_stub(rpc_channel);  // CoordinatorService
    rpc_stub.FLService(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
    fut.wait();
  }
  VLOG(0) << "fl-ps > PushFLClientInfoSync finished, client id: " << _client_id;
  return;
}

std::string BrpcPsClient::PullFlStrategy() {
  while (!_service._is_fl_strategy_ready) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    VLOG(0) << "fl-ps > waiting for fl strategy returned from coordinator";
  }
  _service._is_fl_strategy_ready =
      false;  // only support single thread, no need for multi-threads
  return _service._fl_strategy;
}

int32_t BrpcPsClient::Initialize() {
  _async_call_num = 0;

  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.timeout_ms = FLAGS_pserver_timeout_ms;
  options.connection_type = "pooled";
  options.connect_timeout_ms = FLAGS_pserver_connect_timeout_ms;
  options.max_retry = 3;

  std::ostringstream os;
  std::string server_ip_port;
  std::string client_ip(butil::my_ip_cstr());

  // 获取server列表，并连接
  std::vector<PSHost> server_list = _env->GetPsServers();
  _server_channels.resize(server_list.size());
  for (size_t i = 0; i < server_list.size(); ++i) {
    server_ip_port.assign(server_list[i].ip.c_str());
    server_ip_port.append(":");
    server_ip_port.append(std::to_string(server_list[i].port));
    for (size_t j = 0; j < _server_channels[i].size(); ++j) {
      _server_channels[i][j].reset(new brpc::Channel());
      if (_server_channels[i][j]->Init(server_ip_port.c_str(), "", &options) !=
          0) {
        VLOG(0) << "BrpcPSclient connect to Server:" << server_ip_port
                << " Failed! Try again.";
        std::string int_ip_port =
            GetIntTypeEndpoint(server_list[i].ip, server_list[i].port);
        if (_server_channels[i][j]->Init(int_ip_port.c_str(), "", &options) !=
            0) {
          LOG(ERROR) << "BrpcPSclient connect to Server:" << int_ip_port
                     << " Failed!";
          return -1;
        }
      }
    }
    os << server_ip_port << ",";
  }
  // 启动client探听接口, 并相互建立连接
  StartClientService();

  // 异步push 请求队列初始化
  const auto &worker_param = _config.worker_param().downpour_worker_param();
  for (int i = 0; i < worker_param.downpour_table_param_size(); ++i) {
    auto type = worker_param.downpour_table_param(i).type();
    auto table_id = worker_param.downpour_table_param(i).table_id();
    if (type == PS_DENSE_TABLE) {
      _push_dense_task_queue_map[table_id] =
          ::paddle::framework::MakeChannel<DenseAsyncTask *>();
    }
    if (type == PS_SPARSE_TABLE) {
      _push_sparse_task_queue_map[table_id] =
          ::paddle::framework::MakeChannel<SparseAsyncTask *>();
      _push_sparse_merge_count_map[table_id] = 0;
    }
  }

  auto &profiler = CostProfiler::instance();
  profiler.register_profiler("pserver_client_pull_dense");
  profiler.register_profiler("pserver_client_pull_sparse");
  profiler.register_profiler("pserver_client_pull_sparse_param");
  profiler.register_profiler("pserver_client_pull_sparse_local");
  profiler.register_profiler("pserver_client_push_sparse");
  profiler.register_profiler("pserver_client_push_sparse_parse");
  profiler.register_profiler("client_push_sparse_put");
  profiler.register_profiler("pserver_client_push_sparse");
  profiler.register_profiler("pserver_client_push_sparse_merge");
  profiler.register_profiler("pserver_client_push_sparse_rpc");
  profiler.register_profiler("pserver_client_push_dense");
  profiler.register_profiler("pserver_client_push_dense_parse");
  profiler.register_profiler("push_dense_put");
  profiler.register_profiler("pserver_client_push_dense_merge");
  profiler.register_profiler("pserver_client_push_dense_rpc");
  profiler.register_profiler("pserver_client_push_dense_send");

  _running = true;
  _flushing = false;
  // 启动异步push线程
  _async_push_sparse_thread =
      std::thread(std::bind(&BrpcPsClient::PushSparseTaskConsume, this));
  // _async_push_sparse_thread.detach();
  _async_push_dense_thread =
      std::thread(std::bind(&BrpcPsClient::PushDenseTaskConsume, this));
  // for debug
  // _print_thread =
  //    std::thread(std::bind(&BrpcPsClient::PrintQueueSizeThread, this));

  return 0;
}

int DownpourBrpcClosure::check_response(size_t request_idx, int cmd_id) {
  if (_cntls[request_idx]->Failed()) {
    LOG(ERROR) << "request cmd_id:" << cmd_id
               << " failed, "
                  "err:"
               << _cntls[request_idx]->ErrorText();
    return -1;
  }
  if (_responses[request_idx].err_code() != 0) {
    LOG(ERROR) << "response ret bad, server_idx:" << request_idx
               << "cmd_id:" << cmd_id
               << " err_code:" << _responses[request_idx].err_code()
               << " err_msg:" << _responses[request_idx].err_msg();
    return -1;
  }
  return 0;
}

int DownpourBrpcClosure::check_save_response(size_t request_idx, int cmd_id) {
  int32_t feasign_size = 0;
  if (_cntls[request_idx]->Failed()) {
    LOG(ERROR) << "request cmd_id:" << cmd_id
               << " failed, "
                  "err:"
               << _cntls[request_idx]->ErrorText();
    return -1;
  }
  feasign_size = _responses[request_idx].err_code();
  if (feasign_size < 0) {
    LOG(ERROR) << "response ret bad, server_idx:" << request_idx
               << "cmd_id:" << cmd_id
               << " err_code:" << _responses[request_idx].err_code()
               << " err_msg:" << _responses[request_idx].err_msg();
    return -1;
  }
  return feasign_size;
}

std::string DownpourBrpcClosure::get_response(size_t request_idx, int cmd_id) {
  std::string data = _responses[request_idx].data();
  return data;
}

int FlClientBrpcClosure::check_response(size_t request_idx, int cmd_id) {
  if (_cntls[request_idx]->Failed()) {
    LOG(ERROR) << "request cmd_id:" << cmd_id
               << " failed, "
                  "err:"
               << _cntls[request_idx]->ErrorText();
    return -1;
  }
  if (_responses[request_idx].err_code() != 0) {
    LOG(ERROR) << "response ret bad, server_idx:" << request_idx
               << "cmd_id:" << cmd_id
               << " err_code:" << _responses[request_idx].err_code()
               << " err_msg:" << _responses[request_idx].err_msg();
    return -1;
  }
  return 0;
}

std::future<int32_t> BrpcPsClient::PrintTableStat(uint32_t table_id,
                                                  uint16_t pass_id,
                                                  size_t threshold) {
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [request_call_num, table_id](void *done) {
        int ret = 0;
        uint64_t feasign_size = 0;
        uint64_t mf_size = 0;
        ::paddle::framework::BinaryArchive ar;
        auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PRINT_TABLE_STAT) != 0) {
            ret = -1;
            break;
          }
          std::string resp = closure->get_response(i, PS_PRINT_TABLE_STAT);
          ar.SetReadBuffer(
              const_cast<char *>(resp.c_str()), resp.length(), nullptr);

          feasign_size += ar.Get<uint64_t>();
          mf_size += ar.Get<uint64_t>();
        }
        closure->set_promise_value(ret);
        std::cout << "table id: " << table_id
                  << ", feasign size: " << feasign_size
                  << ", mf size: " << mf_size << std::endl;
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PRINT_TABLE_STAT);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    PsService_Stub rpc_stub(GetCmdChannel(i));
    closure->cntl(i)->set_timeout_ms(
        10800000);  // cmd msg don't limit timeout for save/load
    rpc_stub.service(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
  }
  return fut;
}
std::future<int32_t> BrpcPsClient::SendCmd(
    uint32_t table_id, int cmd_id, const std::vector<std::string> &params) {
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [request_call_num, cmd_id](void *done) {
        int ret = 0;
        auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, cmd_id) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(cmd_id);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    for (const auto &param : params) {
      closure->request(i)->add_params(param);
    }
    PsService_Stub rpc_stub(GetCmdChannel(i));
    closure->cntl(i)->set_timeout_ms(
        10800000 * 2);  // cmd msg don't limit timeout for save/load
    rpc_stub.service(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::SendSaveCmd(
    uint32_t table_id, int cmd_id, const std::vector<std::string> &params) {
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [request_call_num, cmd_id](void *done) {
        int ret = 0;
        uint32_t feasign_size = 0;
        auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_save_response(i, cmd_id) < 0) {
            ret = -1;
            break;
          }
          feasign_size += closure->check_save_response(i, cmd_id);
        }
        if (ret == 0) {
          closure->set_promise_value(feasign_size);
        } else {
          closure->set_promise_value(ret);
        }
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(cmd_id);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    for (const auto &param : params) {
      closure->request(i)->add_params(param);
    }
    PsService_Stub rpc_stub(GetCmdChannel(i));
    closure->cntl(i)->set_timeout_ms(
        10800000);  // cmd msg don't limit timeout for save/load
    rpc_stub.service(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::Shrink(uint32_t table_id,
                                          const std::string threshold) {
  return SendCmd(table_id, PS_SHRINK_TABLE, {threshold});
}

std::future<int32_t> BrpcPsClient::Load(const std::string &epoch,
                                        const std::string &mode) {
  return SendCmd(-1, PS_LOAD_ALL_TABLE, {epoch, mode});
}
std::future<int32_t> BrpcPsClient::Load(uint32_t table_id,
                                        const std::string &epoch,
                                        const std::string &mode) {
  return SendCmd(table_id, PS_LOAD_ONE_TABLE, {epoch, mode});
}

std::future<int32_t> BrpcPsClient::Save(const std::string &epoch,
                                        const std::string &mode) {
  VLOG(1) << "BrpcPsClient::save path " << epoch;
  return SendSaveCmd(-1, PS_SAVE_ALL_TABLE, {epoch, mode});
}
std::future<int32_t> BrpcPsClient::Save(uint32_t table_id,
                                        const std::string &epoch,
                                        const std::string &mode) {
  VLOG(1) << "BrpcPsClient::save one table path " << epoch << " table_id "
          << table_id;
  return SendSaveCmd(table_id, PS_SAVE_ONE_TABLE, {epoch, mode});
}

std::future<int32_t> BrpcPsClient::CacheShuffle(
    uint32_t table_id,
    const std::string &path,
    const std::string &mode,
    const std::string &cache_threshold) {
  VLOG(1) << "BrpcPsClient send cmd for cache shuffle";
  return SendSaveCmd(table_id, PS_CACHE_SHUFFLE, {path, mode, cache_threshold});
}

std::future<int32_t> BrpcPsClient::CacheShuffleMultiTable(
    std::vector<int> tables,
    const std::string &path,
    const std::string &mode,
    const std::string &cache_threshold) {
  VLOG(1) << "BrpcPsClient send cmd for cache shuffle multi table one path";
  std::vector<std::string> param;
  param.push_back(path);
  param.push_back(mode);
  param.push_back(cache_threshold);
  for (auto table : tables) {
    param.push_back(std::to_string(table));
  }
  return SendSaveCmd(0, PS_CACHE_SHUFFLE, param);
}

std::future<int32_t> BrpcPsClient::SaveCache(uint32_t table_id,
                                             const std::string &path,
                                             const std::string &mode) {
  return SendSaveCmd(table_id, PS_SAVE_ONE_CACHE_TABLE, {path, mode});
}

std::future<int32_t> BrpcPsClient::GetCacheThreshold(uint32_t table_id,
                                                     double &cache_threshold) {
  int cmd_id = PS_GET_CACHE_THRESHOLD;
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num,
      [request_call_num, cmd_id, &cache_threshold](void *done) {
        int ret = 0;
        auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
        std::vector<double> cache_thresholds(request_call_num, 0);
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, cmd_id) != 0) {
            ret = -1;
            break;
          }
          std::string cur_res = closure->get_response(i, cmd_id);
          cache_thresholds[i] = std::stod(cur_res);
        }
        double sum_threshold = 0.0;
        int count = 0;
        for (auto t : cache_thresholds) {
          if (t >= 0) {
            sum_threshold += t;
            ++count;
          }
        }
        if (count == 0) {
          cache_threshold = 0;
        } else {
          cache_threshold = sum_threshold / count;
        }
        VLOG(1) << "client get cache threshold: " << cache_threshold;
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(cmd_id);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    PsService_Stub rpc_stub(GetCmdChannel(i));
    closure->cntl(i)->set_timeout_ms(10800000);
    rpc_stub.service(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::Clear() {
  return SendCmd(-1, PS_CLEAR_ALL_TABLE, {});
}
std::future<int32_t> BrpcPsClient::Clear(uint32_t table_id) {
  return SendCmd(table_id, PS_CLEAR_ONE_TABLE, {});
}

std::future<int32_t> BrpcPsClient::Revert() {
  return SendCmd(-1, PS_REVERT, {});
}

std::future<int32_t> BrpcPsClient::CheckSavePrePatchDone() {
  return SendCmd(-1, PS_CHECK_SAVE_PRE_PATCH_DONE, {});
}

std::future<int32_t> BrpcPsClient::Flush() {
  VLOG(0) << "BrpcPsClient::flush begin";
  _flushing = true;
  std::promise<int> promise;
  std::future<int32_t> fut = promise.get_future();
  do {
    VLOG(3) << "wait _async_call_num:" << _async_call_num;
    usleep(100000);  // sleep 100ms wait async end
  } while (_async_call_num > 0);
  VLOG(1) << "flush _async_call_num = 0";
  promise.set_value(0);
  _flushing = false;
  VLOG(0) << "BrpcPsClient::flush done";
  PrintQueueSize();
  return fut;
}

void BrpcPsClient::PrintQueueSize() {
  for (auto &push_sparse_task_itr : _push_sparse_task_queue_map) {
    auto table_id = push_sparse_task_itr.first;
    auto queue_size = push_sparse_task_itr.second->Size();
    VLOG(0) << "BrpcPsClient::PrintQueueSize: table " << table_id
            << " size: " << queue_size;
  }

  for (auto &task_queue_itr : _push_dense_task_queue_map) {
    auto table_id = task_queue_itr.first;
    auto queue_size = task_queue_itr.second->Size();
    VLOG(0) << "BrpcPsClient::PrintQueueSize: table " << table_id
            << " size: " << queue_size;
  }
}

void BrpcPsClient::PrintQueueSizeThread() {
  while (_running) {
    usleep(1000000 * 60 * 2);
    PrintQueueSize();
  }
}

void BrpcPsClient::FinalizeWorker() {
  Flush();
  VLOG(0) << "BrpcPsClient::FinalizeWorker begin join thread";
  _running = false;
  _async_push_dense_thread.join();
  _async_push_sparse_thread.join();
  // _print_thread.join();
  VLOG(0) << "BrpcPsClient::FinalizeWorker begin join server";
  _server.Stop(1000);
  _server.Join();
  _server_started = false;
  VLOG(0) << "BrpcPsClient::FinalizeWorker done";
}

std::future<int32_t> BrpcPsClient::StopServer() {
  return SendCmd(-1, PS_STOP_SERVER, {});
}

std::future<int32_t> BrpcPsClient::StartProfiler() {
  return SendCmd(-1, PS_START_PROFILER, {});
}

std::future<int32_t> BrpcPsClient::StopProfiler() {
  return SendCmd(-1, PS_STOP_PROFILER, {});
}

std::future<int32_t> BrpcPsClient::Barrier(size_t table_id,
                                           uint32_t barrier_type) {
  return SendCmd(table_id, PS_BARRIER, {std::to_string(barrier_type)});
}

std::future<int32_t> BrpcPsClient::PullGeoParam(size_t table_id,
                                                std::vector<float> *values,
                                                std::vector<uint64_t> *keys,
                                                int pserver_idx) {
  auto *accessor = GetTableAccessor(table_id);
  DownpourBrpcClosure *closure =
      new DownpourBrpcClosure(1, [keys, values, accessor](void *done) {
        int ret = 0;
        auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
        uint32_t shard_nums;
        if (closure->check_response(0, PS_PULL_GEO_PARAM) != 0) {
          ret = -1;
        }
        auto &res_io_buffer = closure->cntl(0)->response_attachment();
        butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
        io_buffer_itr.copy_and_forward(reinterpret_cast<void *>(&shard_nums),
                                       sizeof(uint32_t));
        keys->resize(shard_nums);
        values->resize(shard_nums * accessor->GetAccessorInfo().update_dim);
        io_buffer_itr.copy_and_forward((void *)(keys->data()),  // NOLINT
                                       sizeof(uint64_t) * shard_nums);
        io_buffer_itr.copy_and_forward(
            (void *)(values->data()),  // NOLINT
            shard_nums * accessor->GetAccessorInfo().update_size);
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  closure->request(0)->set_cmd_id(PS_PULL_GEO_PARAM);
  closure->request(0)->set_table_id(table_id);
  closure->request(0)->set_client_id(_client_id);
  PsService_Stub rpc_stub(GetCmdChannel(pserver_idx));
  closure->cntl(0)->set_log_id(butil::gettimeofday_ms());
  rpc_stub.service(
      closure->cntl(0), closure->request(0), closure->response(0), closure);
  return fut;
}

// for GEO
std::future<int32_t> BrpcPsClient::PushSparseParam(size_t table_id,
                                                   const uint64_t *keys,
                                                   const float **update_values,
                                                   size_t num,
                                                   void *done) {
  auto *accessor = GetTableAccessor(table_id);
  // 发送RPC请求
  DownpourBrpcClosure *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  size_t request_call_num = _server_channels.size();
  std::vector<std::vector<uint64_t>> ids;
  std::vector<std::vector<const float *>> value_ptrs;
  ids.resize(request_call_num);
  value_ptrs.resize(request_call_num);

  for (size_t i = 0; i < num; ++i) {
    size_t pserver_idx = keys[i] % request_call_num;
    ids[pserver_idx].push_back(keys[i]);
    value_ptrs[pserver_idx].push_back(update_values[i]);
  }
  for (size_t shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
    auto kvs = ids[shard_idx];
    auto value_ptr = value_ptrs[shard_idx];
    size_t kv_size = kvs.size();
    uint32_t value_size = accessor->GetAccessorInfo().update_size;
    // 发送RPC请求
    auto *push_request = closure->request(shard_idx);
    push_request->set_cmd_id(PS_PUSH_SPARSE_PARAM);
    push_request->set_table_id(table_id);
    push_request->set_client_id(_client_id);
    push_request->add_params((char *)&kv_size, sizeof(uint32_t));  // NOLINT
    auto *push_data = push_request->mutable_data();
    push_data->resize(kv_size * (sizeof(uint64_t) + value_size));
    char *push_data_ptr = const_cast<char *>(push_data->data());
    memcpy(push_data_ptr, kvs.data(), kv_size * sizeof(uint64_t));
    push_data_ptr += kv_size * sizeof(uint64_t);
    for (size_t i = 0; i < kv_size; ++i) {
      memcpy(push_data_ptr, value_ptr[i], value_size);
      push_data_ptr += value_size;
    }
    PsService_Stub rpc_stub(GetSparseChannel(shard_idx));
    closure->cntl(shard_idx)->set_request_compress_type(
        (brpc::CompressType)FLAGS_pserver_communicate_compress_type);
    rpc_stub.service(closure->cntl(shard_idx),
                     closure->request(shard_idx),
                     closure->response(shard_idx),
                     closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::PullDense(Region *regions,
                                             size_t region_num,
                                             size_t table_id) {
  auto timer = std::make_shared<CostTimer>("pserver_client_pull_dense");
  auto *accessor = GetTableAccessor(table_id);
  auto fea_dim = accessor->GetAccessorInfo().fea_dim;
  size_t request_call_num = _server_channels.size();
  uint32_t num_per_shard = DenseDimPerShard(fea_dim, request_call_num);
  // callback 将各shard结果，顺序填入region
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num,
      [request_call_num, num_per_shard, regions, region_num, accessor](
          void *done) {
        int ret = 0;
        size_t region_idx = 0;       // 当前填充的region偏移
        size_t region_data_idx = 0;  // 当前填充的region内data偏移
        auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
        size_t shard_data_size =
            num_per_shard * accessor->GetAccessorInfo().select_size;
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PULL_DENSE_TABLE) != 0) {
            ret = -1;
            break;
          }
          auto &res_io_buffer = closure->cntl(i)->response_attachment();

          butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
          size_t shard_buffer_remain = res_io_buffer.size();
          if (shard_buffer_remain != shard_data_size) {
            LOG(ERROR) << "expect res_size:" << shard_data_size
                       << ", but size:" << shard_buffer_remain
                       << ", ignore this response";
            ret = -1;
            break;
          }
          while (shard_buffer_remain > 0 && region_idx < region_num) {
            auto &region = regions[region_idx];
            if (region.size - region_data_idx >= shard_buffer_remain) {
              // region待填充空间 >= 分片buffer数据, 直接拷贝置入
              io_buffer_itr.copy_and_forward(
                  reinterpret_cast<void *>(region.data + region_data_idx),
                  shard_buffer_remain);
              region_data_idx += shard_buffer_remain;
              shard_buffer_remain = 0;
            } else if (region.size - region_data_idx == 0) {
              // region填满，切换到下一个region
              ++region_idx;
              region_data_idx = 0;
            } else {
              // region不足以容纳所有数据，则能放多少 拷贝多少
              io_buffer_itr.copy_and_forward(
                  reinterpret_cast<void *>(region.data + region_data_idx),
                  region.size - region_data_idx);
              shard_buffer_remain -= (region.size - region_data_idx);
              ++region_idx;
              region_data_idx = 0;
            }
          }
        }
        closure->set_promise_value(ret);
      });
  closure->add_timer(timer);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PULL_DENSE_TABLE);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    closure->request(i)->add_params((char *)&num_per_shard,  // NOLINT
                                    sizeof(num_per_shard));
    PsService_Stub rpc_stub(GetDenseChannel(i));
    rpc_stub.service(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::PushDenseParam(const Region *regions,
                                                  size_t region_num,
                                                  size_t table_id) {
  auto *accessor = GetTableAccessor(table_id);
  auto accessor_info = accessor->GetAccessorInfo();
  size_t request_call_num = _server_channels.size();
  // 1.拆分Region数据到shard中，后续多shard并行拷贝数据
  std::vector<std::vector<Region>> regions_partition(request_call_num);
  uint32_t num_per_shard =
      DenseDimPerShard(accessor_info.fea_dim, request_call_num);
  size_t shard_data_size = num_per_shard * accessor_info.update_size;
  size_t current_region_idx = 0;
  size_t current_region_data_idx = 0;
  for (size_t i = 0; i < request_call_num; ++i) {
    size_t shard_data_remain_size = shard_data_size;
    while (shard_data_remain_size > 0 && current_region_idx < region_num) {
      const auto &region = regions[current_region_idx];
      size_t region_remain_size = region.size - current_region_data_idx;
      if (shard_data_remain_size >= region_remain_size) {
        regions_partition[i].push_back(
            Region(region.data + current_region_data_idx, region_remain_size));
        ++current_region_idx;
        current_region_data_idx = 0;
        shard_data_remain_size -= region_remain_size;
      } else {
        regions_partition[i].push_back(Region(
            region.data + current_region_data_idx, shard_data_remain_size));
        current_region_data_idx += shard_data_remain_size;
        shard_data_remain_size = 0;
      }
    }
  }

  DownpourBrpcClosure *closure =
      new DownpourBrpcClosure(request_call_num, [request_call_num](void *done) {
        int ret = 0;
        auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PUSH_DENSE_PARAM) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  static const int REGION_ASSIGN_BUFFER_SIZE = 1024 * 10;
  // 用于数据补齐
  static char region_assign_buffer[REGION_ASSIGN_BUFFER_SIZE];  // NOLINT
  // 开始多shard并行拷贝&请求
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PUSH_DENSE_PARAM);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    auto &request_buffer = closure->cntl(i)->request_attachment();
    request_buffer.append(reinterpret_cast<void *>(&num_per_shard),
                          sizeof(uint32_t));
    auto &region_list = regions_partition[i];
    size_t fill_remain_size = shard_data_size;
    for (auto &region : region_list) {
      fill_remain_size -= region.size;
      request_buffer.append(reinterpret_cast<void *>(region.data), region.size);
    }
    // 保证各分片数据对齐
    while (fill_remain_size > 0) {
      size_t fill_num = fill_remain_size > REGION_ASSIGN_BUFFER_SIZE
                            ? REGION_ASSIGN_BUFFER_SIZE
                            : fill_remain_size;
      request_buffer.append(reinterpret_cast<void *>(region_assign_buffer),
                            fill_num);
      fill_remain_size -= fill_num;
    }
    PsService_Stub rpc_stub(GetDenseChannel(i));
    rpc_stub.service(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::PushSparseRawGradient(
    size_t table_id,
    const uint64_t *keys,
    const float **update_values,
    size_t num,
    void *done) {
  auto *accessor = GetTableAccessor(table_id);
  // 发送RPC请求
  DownpourBrpcClosure *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  size_t request_call_num = _server_channels.size();
  std::vector<std::vector<uint64_t>> ids;
  std::vector<std::vector<const float *>> value_ptrs;
  ids.resize(request_call_num);
  value_ptrs.resize(request_call_num);

  const auto &server_param = _config.server_param().downpour_server_param();
  uint64_t shard_num = FLAGS_pserver_sparse_table_shard_num;
  for (int i = 0; i < server_param.downpour_table_param_size(); ++i) {
    const auto &table_param = server_param.downpour_table_param(i);
    if (table_param.table_id() == table_id) {
      shard_num = table_param.shard_num();
      break;
    }
  }

  for (size_t i = 0; i < num; ++i) {
    size_t pserver_idx = get_sparse_shard(shard_num, request_call_num, keys[i]);
    ids[pserver_idx].push_back(keys[i]);
    value_ptrs[pserver_idx].push_back(update_values[i]);
  }

  for (size_t shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
    auto kvs = ids[shard_idx];
    auto value_ptr = value_ptrs[shard_idx];

    size_t kv_size = kvs.size();
    uint32_t value_size = accessor->GetAccessorInfo().update_size;

    // 发送RPC请求
    auto *push_request = closure->request(shard_idx);
    push_request->set_cmd_id(PS_PUSH_SPARSE_TABLE);
    push_request->set_table_id(table_id);
    push_request->set_client_id(_client_id);
    push_request->add_params((char *)&kv_size, sizeof(uint32_t));  // NOLINT
    auto *push_data = push_request->mutable_data();
    push_data->resize(kv_size * (sizeof(uint64_t) + value_size));
    char *push_data_ptr = const_cast<char *>(push_data->data());
    memcpy(push_data_ptr, kvs.data(), kv_size * sizeof(uint64_t));
    push_data_ptr += kv_size * sizeof(uint64_t);

    for (size_t i = 0; i < kv_size; ++i) {
      memcpy(push_data_ptr, value_ptr[i], value_size);
      push_data_ptr += value_size;
    }
    PsService_Stub rpc_stub(GetSparseChannel(shard_idx));
    closure->cntl(shard_idx)->set_request_compress_type(
        (brpc::CompressType)FLAGS_pserver_communicate_compress_type);
    rpc_stub.service(closure->cntl(shard_idx),
                     closure->request(shard_idx),
                     closure->response(shard_idx),
                     closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::PushDenseRawGradient(
    int table_id,
    float *total_send_data,
    size_t total_send_data_size,
    void *done) {
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  auto *accessor = GetTableAccessor(table_id);
  uint32_t num_per_shard =
      DenseDimPerShard(accessor->GetAccessorInfo().fea_dim, request_call_num);
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PUSH_DENSE_TABLE);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    auto *push_data = closure->request(i)->mutable_data();
    push_data->clear();
    push_data->resize(sizeof(uint32_t) + num_per_shard * sizeof(float));
    char *push_data_ptr = const_cast<char *>(push_data->data());
    memcpy(push_data_ptr, &num_per_shard, sizeof(uint32_t));
    memcpy(push_data_ptr + sizeof(uint32_t),
           total_send_data + i * num_per_shard,
           num_per_shard * sizeof(float));
    // closure->cntl(i)->set_request_compress_type(
    //     (brpc::CompressType)FLAGS_pserver_communicate_compress_type);
    PsService_Stub rpc_stub(GetDenseChannel(i));
    rpc_stub.service(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::PushGlobalStep(int table_id,
                                                  int64_t *total_send_data,
                                                  void *done) {
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PUSH_GLOBAL_STEP);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    auto *push_data = closure->request(i)->mutable_data();
    push_data->clear();
    int32_t num_per_shard = 1;
    push_data->resize(sizeof(uint32_t) + num_per_shard * sizeof(int64_t));
    char *push_data_ptr = const_cast<char *>(push_data->data());
    memcpy(push_data_ptr, &num_per_shard, sizeof(uint32_t));
    memcpy(push_data_ptr + sizeof(uint32_t),
           total_send_data,
           num_per_shard * sizeof(int64_t));

    PsService_Stub rpc_stub(GetDenseChannel(i));
    rpc_stub.service(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::PullSparse(float **select_values,
                                              size_t table_id,
                                              const uint64_t *keys,
                                              size_t num,
                                              bool is_training) {
  auto timer = std::make_shared<CostTimer>("pserver_client_pull_sparse");
  auto local_timer =
      std::make_shared<CostTimer>("pserver_client_pull_sparse_local");
  size_t request_call_num = _server_channels.size();

  auto shard_sorted_kvs = std::make_shared<
      std::vector<std::vector<std::pair<uint64_t, float *>>>>();
  shard_sorted_kvs->resize(request_call_num);

  const auto &server_param = _config.server_param().downpour_server_param();
  uint64_t shard_num = FLAGS_pserver_sparse_table_shard_num;
  for (int i = 0; i < server_param.downpour_table_param_size(); ++i) {
    const auto &table_param = server_param.downpour_table_param(i);
    if (table_param.table_id() == table_id) {
      shard_num = table_param.shard_num();
      break;
    }
  }

  for (size_t i = 0; i < num; ++i) {
    size_t shard_id = get_sparse_shard(shard_num, request_call_num, keys[i]);
    shard_sorted_kvs->at(shard_id).push_back({keys[i], select_values[i]});
  }

  auto *accessor = GetTableAccessor(table_id);

  size_t value_size = accessor->GetAccessorInfo().select_size;

  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [shard_sorted_kvs, value_size](void *done) {
        int ret = 0;
        auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
        for (size_t i = 0; i < shard_sorted_kvs->size(); ++i) {
          if (closure->check_response(i, PS_PULL_SPARSE_TABLE) != 0) {
            ret = -1;
            break;
          }

          auto &request_kvs = shard_sorted_kvs->at(i);
          auto &res_io_buffer = closure->cntl(i)->response_attachment();
          butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
          uint64_t last_key = UINT64_MAX;
          float *last_value_data = NULL;

          for (auto &kv_pair : request_kvs) {
            if (kv_pair.first == last_key) {
              memcpy(reinterpret_cast<void *>(kv_pair.second),
                     reinterpret_cast<void *>(last_value_data),
                     value_size);
            } else {
              last_key = kv_pair.first;
              last_value_data = kv_pair.second;
              if (value_size !=
                  io_buffer_itr.copy_and_forward(
                      reinterpret_cast<void *>(last_value_data), value_size)) {
                LOG(WARNING) << "res data is lack or not in format";
                ret = -1;
                break;
              }
            }
          }
        }
        closure->set_promise_value(ret);
      });
  closure->add_timer(timer);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  for (size_t i = 0; i < request_call_num; ++i) {
    auto &sorted_kvs = shard_sorted_kvs->at(i);
    std::sort(sorted_kvs.begin(),
              sorted_kvs.end(),
              [](const std::pair<uint64_t, float *> &k1,
                 const std::pair<uint64_t, float *> &k2) {
                return k1.first < k2.first;
              });

    uint64_t last_key = UINT64_MAX;
    uint32_t kv_request_count = 0;
    size_t sorted_kv_size = sorted_kvs.size();
    auto &request_buffer = closure->cntl(i)->request_attachment();

    request_buffer.append(reinterpret_cast<void *>(&is_training), sizeof(bool));
    std::vector<uint32_t> keys_counter;
    keys_counter.reserve(sorted_kv_size);

    for (size_t kv_idx = 0; kv_idx < sorted_kv_size; ++kv_idx) {
      ++kv_request_count;
      uint32_t keys = 1;
      last_key = sorted_kvs[kv_idx].first;
      request_buffer.append(reinterpret_cast<void *>(&last_key),
                            sizeof(uint64_t));
      while (kv_idx < sorted_kv_size - 1 &&
             last_key == sorted_kvs[kv_idx + 1].first) {
        ++kv_idx;
        ++keys;
      }
      keys_counter.push_back(keys);
    }

    request_buffer.append(reinterpret_cast<void *>(keys_counter.data()),
                          sizeof(uint32_t) * keys_counter.size());

    if (kv_request_count == 0) {
      closure->Run();
    } else {
      closure->request(i)->set_cmd_id(PS_PULL_SPARSE_TABLE);
      closure->request(i)->set_table_id(table_id);
      closure->request(i)->set_client_id(_client_id);
      closure->request(i)->add_params((char *)&kv_request_count,  // NOLINT
                                      sizeof(uint32_t));
      PsService_Stub rpc_stub(GetCmdChannel(i));
      closure->cntl(i)->set_log_id(butil::gettimeofday_ms());
      rpc_stub.service(
          closure->cntl(i), closure->request(i), closure->response(i), closure);
    }
  }
  return fut;
}

// for GEO
std::future<int32_t> BrpcPsClient::PullSparseParam(float **select_values,
                                                   size_t table_id,
                                                   const uint64_t *keys,
                                                   size_t num,
                                                   bool is_training) {
  auto timer = std::make_shared<CostTimer>("pserver_client_pull_sparse_param");
  size_t request_call_num = _server_channels.size();

  auto shard_sorted_kvs = std::make_shared<
      std::vector<std::vector<std::pair<uint64_t, float *>>>>();
  shard_sorted_kvs->resize(request_call_num);

  for (size_t i = 0; i < num; ++i) {
    size_t shard_id = keys[i] % request_call_num;
    shard_sorted_kvs->at(shard_id).push_back({keys[i], select_values[i]});
  }

  auto *accessor = GetTableAccessor(table_id);
  size_t value_size = accessor->GetAccessorInfo().select_size;
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [shard_sorted_kvs, value_size](void *done) {
        int ret = 0;
        auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
        for (size_t i = 0; i < shard_sorted_kvs->size(); ++i) {
          if (closure->check_response(i, PS_PULL_SPARSE_TABLE) != 0) {
            ret = -1;
            break;
          }

          auto &request_kvs = shard_sorted_kvs->at(i);
          auto &res_io_buffer = closure->cntl(i)->response_attachment();
          butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
          uint64_t last_key = UINT64_MAX;
          float *last_value_data = NULL;

          // can remove sort&unique
          for (auto &kv_pair : request_kvs) {
            if (kv_pair.first == last_key) {
              memcpy(reinterpret_cast<void *>(kv_pair.second),
                     reinterpret_cast<void *>(last_value_data),
                     value_size);
            } else {
              last_key = kv_pair.first;
              last_value_data = kv_pair.second;
              if (value_size !=
                  io_buffer_itr.copy_and_forward(
                      reinterpret_cast<void *>(last_value_data), value_size)) {
                LOG(WARNING) << "res data is lack or not in format";
                ret = -1;
                break;
              }
            }
          }
        }
        closure->set_promise_value(ret);
      });

  closure->add_timer(timer);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  for (size_t i = 0; i < request_call_num; ++i) {
    auto &sorted_kvs = shard_sorted_kvs->at(i);
    std::sort(sorted_kvs.begin(),
              sorted_kvs.end(),
              [](const std::pair<uint64_t, float *> &k1,
                 const std::pair<uint64_t, float *> &k2) {
                return k1.first < k2.first;
              });

    uint64_t last_key = UINT64_MAX;
    uint32_t kv_request_count = 0;
    size_t sorted_kv_size = sorted_kvs.size();
    auto &request_buffer = closure->cntl(i)->request_attachment();

    request_buffer.append(reinterpret_cast<void *>(&is_training), sizeof(bool));
    std::vector<uint32_t> keys_counter;
    keys_counter.reserve(sorted_kv_size);

    for (size_t kv_idx = 0; kv_idx < sorted_kv_size; ++kv_idx) {
      ++kv_request_count;
      uint32_t keys = 1;
      last_key = sorted_kvs[kv_idx].first;
      request_buffer.append(reinterpret_cast<void *>(&last_key),
                            sizeof(uint64_t));
      while (kv_idx < sorted_kv_size - 1 &&
             last_key == sorted_kvs[kv_idx + 1].first) {
        ++kv_idx;
        ++keys;
      }
      keys_counter.push_back(keys);
    }

    request_buffer.append(reinterpret_cast<void *>(keys_counter.data()),
                          sizeof(uint32_t) * keys_counter.size());

    if (kv_request_count == 0) {
      closure->Run();
    } else {
      closure->request(i)->set_cmd_id(PS_PULL_SPARSE_TABLE);
      closure->request(i)->set_table_id(table_id);
      closure->request(i)->set_client_id(_client_id);
      closure->request(i)->add_params((char *)&kv_request_count,  // NOLINT
                                      sizeof(uint32_t));
      PsService_Stub rpc_stub(GetCmdChannel(i));
      closure->cntl(i)->set_log_id(butil::gettimeofday_ms());
      rpc_stub.service(
          closure->cntl(i), closure->request(i), closure->response(i), closure);
    }
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::SendClient2ClientMsg(
    int msg_type, int to_client_id, const std::string &msg) {
  auto promise = std::make_shared<std::promise<int32_t>>();
  std::future<int> fut = promise->get_future();
  if (to_client_id >= 0 &&
      static_cast<size_t>(to_client_id) >= _client_channels.size()) {
    VLOG(0) << "to_client_id is out of range clients, which size is "
            << _client_channels.size();
    promise->set_value(-1);
    return fut;
  }
  auto *closure = new DownpourBrpcClosure(1, [msg_type](void *done) {
    auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
    int32_t ret = closure->check_response(0, msg_type + 1000);
    closure->set_promise_value(ret);
  });
  closure->add_promise(promise);
  closure->request(0)->set_cmd_id(msg_type);
  closure->request(0)->set_client_id(_client_id);
  closure->request(0)->set_data(msg);
  PsService_Stub rpc_stub(_client_channels[to_client_id].get());
  rpc_stub.service(
      closure->cntl(0), closure->request(0), closure->response(0), closure);
  return fut;
}

std::future<int32_t> BrpcPsClient::PushSparseRawGradientPartial(
    size_t table_id,
    const uint64_t *keys,
    const float **update_values,
    uint32_t num,
    void *done,
    int pserver_idx) {
  auto *accessor = GetTableAccessor(table_id);
  size_t value_size = accessor->GetAccessorInfo().update_size;
  DownpourBrpcClosure *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  // 发送RPC请求
  auto *push_request = closure->request(0);
  push_request->set_cmd_id(PS_PUSH_SPARSE_TABLE);
  push_request->set_table_id(table_id);
  push_request->set_client_id(_client_id);
  push_request->add_params((char *)&num, sizeof(uint32_t));  // NOLINT
  auto *push_data = push_request->mutable_data();
  push_data->resize(num * (sizeof(uint64_t) + value_size));
  char *push_data_ptr = const_cast<char *>(push_data->data());
  memcpy(push_data_ptr, keys, num * sizeof(uint64_t));
  push_data_ptr += num * sizeof(uint64_t);
  for (uint32_t i = 0; i < num; ++i) {
    memcpy(push_data_ptr, update_values[i], value_size);
    push_data_ptr += value_size;
  }
  PsService_Stub rpc_stub(GetSparseChannel(pserver_idx));
  closure->cntl(0)->set_request_compress_type(
      (brpc::CompressType)FLAGS_pserver_communicate_compress_type);
  rpc_stub.service(
      closure->cntl(0), closure->request(0), closure->response(0), closure);
  return fut;
}

int32_t BrpcPsClient::RecvAndSaveTable(const uint64_t table_id,
                                       const std::string &path) {
  // get var information
  std::string var_name = "";
  int64_t var_num = 0;
  int64_t var_shape = 0;
  std::string table_class;
  const auto &worker_param = _config.worker_param().downpour_worker_param();
  for (int i = 0; i < worker_param.downpour_table_param_size(); ++i) {
    if (worker_param.downpour_table_param(i).table_id() == table_id) {
      var_name = worker_param.downpour_table_param(i).common().table_name();
      var_num = worker_param.downpour_table_param(i).common().table_num();
      var_shape = worker_param.downpour_table_param(i).common().table_dim();
      table_class = worker_param.downpour_table_param(i).table_class();
      break;
    }
  }

  PADDLE_ENFORCE_NE(
      var_name,
      "",
      common::errors::InvalidArgument(
          "Cannot find table id %d to save variables.", table_id));

  std::string var_store = string::Sprintf("%s", path);
  MkDirRecursively(var_store.c_str());

  // pull sparse from server
  std::vector<float> save_huge_vec(var_num * var_shape);
  std::vector<uint64_t> save_key(var_num);
  std::vector<float *> save_vec;
  for (size_t i = 0; i < save_key.size(); ++i) {
    save_key[i] = i;
    save_vec.push_back(save_huge_vec.data() + i * var_shape);
  }

  VLOG(2) << "RecvAndSaveTable: table_class: " << table_class;
  // TODO(zhaocaibei123): new GeoBrpcPSClient, move this to its
  // RecvAndSaveTable
  if (table_class == "MemorySparseGeoTable") {
    auto status = PullSparseParam(reinterpret_cast<float **>(save_vec.data()),
                                  table_id,
                                  save_key.data(),
                                  save_key.size(),
                                  true);
    status.wait();
  } else {
    auto status = PullSparse(reinterpret_cast<float **>(save_vec.data()),
                             table_id,
                             save_key.data(),
                             save_key.size(),
                             true);
    status.wait();
  }

  // create lod tensor
  std::shared_ptr<framework::Scope> scope;
  scope.reset(new framework::Scope());
  auto place = phi::CPUPlace();
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(place);

  framework::Variable *var = scope->Var(var_name);
  phi::DenseTensor *var_tensor = var->GetMutable<phi::DenseTensor>();

  std::vector<int64_t> vec_dim = {var_num, var_shape};
  var_tensor->Resize(common::make_ddim(vec_dim));

  // copy and save
  float *tensor_data = var_tensor->mutable_data<float>(place);
  memcpy(
      tensor_data, save_huge_vec.data(), var_num * var_shape * sizeof(float));

  std::string file_name = string::Sprintf("%s/%s", var_store, var_name);
  std::ofstream fout(file_name, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    common::errors::Unavailable(
                        "Cannot open %s to save variables.", file_name));

  phi::SerializeToStream(fout, *var_tensor, dev_ctx);
  fout.close();

  return 0;
}

std::future<int32_t> BrpcPsClient::PushSparse(size_t table_id,
                                              const uint64_t *keys,
                                              const float **update_values,
                                              size_t num) {
  auto push_timer = std::make_shared<CostTimer>("pserver_client_push_sparse");
  CostTimer parse_timer("pserver_client_push_sparse_parse");
  int push_sparse_async_num = _push_sparse_task_queue_map[table_id]->Size();
  while (push_sparse_async_num > FLAGS_pserver_max_async_call_num) {
    //    LOG(INFO) << "PushSparse Waiting for async_call_num consume,
    //    task_num:"
    //              << push_sparse_async_num
    //              << ", max_task_limit:" << FLAGS_pserver_max_async_call_num;
    usleep(5000);  // 5ms
    push_sparse_async_num = _push_sparse_task_queue_map[table_id]->Size();
  }
  auto put_timer = std::make_shared<CostTimer>("client_push_sparse_put");
  thread_local std::vector<std::vector<std::pair<uint64_t, const float *>>>
      shard_sorted_kv_list;
  auto *accessor = GetTableAccessor(table_id);
  size_t request_call_num = _server_channels.size();
  shard_sorted_kv_list.resize(request_call_num);
  for (auto &x : shard_sorted_kv_list) {
    x.clear();
  }
  const auto &server_param = _config.server_param().downpour_server_param();
  uint64_t shard_num = FLAGS_pserver_sparse_table_shard_num;
  for (int i = 0; i < server_param.downpour_table_param_size(); ++i) {
    const auto &table_param = server_param.downpour_table_param(i);
    if (table_param.table_id() == table_id) {
      shard_num = table_param.shard_num();
      break;
    }
  }
  for (size_t i = 0; i < num; ++i) {
    size_t shard_id = get_sparse_shard(shard_num, request_call_num, keys[i]);
    shard_sorted_kv_list[shard_id].push_back({keys[i], update_values[i]});
  }
  auto sparse_task_data = _sparse_task_pool.get();
  sparse_task_data->shared_data.resize(request_call_num);
  auto async_task = new SparseAsyncTask(sparse_task_data, table_id, push_timer);

  for (size_t i = 0; i < request_call_num; ++i) {
    auto &sorted_kv_list = shard_sorted_kv_list[i];
    size_t sorted_kv_size = sorted_kv_list.size();
    auto &shard_kv_data = async_task->data()->shared_data[i];
    shard_kv_data.key_list.resize(sorted_kv_size);
    shard_kv_data.value_list.resize(sorted_kv_size);

    if (sorted_kv_size == 0) {
      shard_kv_data.kv_num = 0;
      continue;
    }
    uint32_t value_size = accessor->GetAccessorInfo().update_size;
    for (size_t kv_idx = 0; kv_idx < sorted_kv_size; ++kv_idx) {
      shard_kv_data.key_list[kv_idx] = sorted_kv_list[kv_idx].first;
      shard_kv_data.value_list[kv_idx].assign(
          (const char *)sorted_kv_list[kv_idx].second, value_size);
    }
    shard_kv_data.kv_num = sorted_kv_size;
  }

  std::future<int> fut = async_task->get_future();
  _push_sparse_task_queue_map[table_id]->Put(std::move(async_task));
  return fut;
}

void BrpcPsClient::PushSparseTaskConsume() {
  uint64_t merge_size = FLAGS_pserver_push_sparse_merge_limit;
  std::vector<std::shared_ptr<SparseAsyncTask>> task_list;
  size_t request_call_num = _server_channels.size();
  ::ThreadPool async_push_sparse_shard_threads(
      FLAGS_pserver_sparse_merge_thread);
  while (_running) {
    auto async_start_time_ms = butil::gettimeofday_ms();
    // 所有sparseTable的pushTask 进行处理
    for (auto &push_sparse_task_itr : _push_sparse_task_queue_map) {
      auto table_id = push_sparse_task_itr.first;
      auto *accessor = GetTableAccessor(table_id);
      auto &task_queue = push_sparse_task_itr.second;
      auto queue_size = task_queue->Size();
      if (queue_size == 0) {
        continue;
      }
      if (merge_size > 0 && (queue_size <= 1 && _flushing == false)) {
        continue;
      }
      ++_async_call_num;

      int merge_count = 0;
      for (auto &task : task_list) {
        if (task->data()) {
          _sparse_task_pool.push(task->data());
        }
      }
      auto sparse_task_data = _sparse_task_pool.get();

      task_list.clear();
      int cur_merge_size = task_queue->Size();

      // task_list[0] 为一个空SparseAsyncTask, 分shard异步merge结果存入此结构。
      sparse_task_data->shared_data.resize(request_call_num);
      auto push_timer =
          std::make_shared<CostTimer>("pserver_client_push_sparse");

      auto async_task =
          new SparseAsyncTask(sparse_task_data, table_id, push_timer);

      task_list.reserve(cur_merge_size + 1);

      task_list.push_back(std::shared_ptr<SparseAsyncTask>(async_task));

      while (!task_queue->Empty() && merge_count < cur_merge_size) {
        ++merge_count;
        SparseAsyncTask *task = nullptr;
        task_queue->Get(task);
        task_list.push_back(std::shared_ptr<SparseAsyncTask>(task));
      }

      _push_sparse_merge_count_map[table_id] += merge_count;

      // 达到或大于 merge_size发送, 发送过程中
      std::vector<int> request_kv_num(request_call_num, 0);

      if (_push_sparse_merge_count_map[table_id] >= merge_size ||
          _flushing == true) {
        DownpourBrpcClosure *closure = new DownpourBrpcClosure(
            request_call_num, [this, request_call_num](void *done) {
              int ret = 0;
              auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
              for (size_t i = 0; i < request_call_num; ++i) {
                if (closure->check_response(i, PS_PUSH_SPARSE_TABLE) != 0) {
                  ret = -1;
                  break;
                }
              }
              closure->set_promise_value(ret);
              --_async_call_num;
            });

        for_each(task_list.begin() + 1,
                 task_list.end(),
                 [closure](std::shared_ptr<SparseAsyncTask> &task) {
                   closure->add_timer(task->timer());
                   closure->add_promise(task->promise());
                 });

        CostTimer merge_timer("pserver_client_push_sparse_merge");
        auto rpc_timer =
            std::make_shared<CostTimer>("pserver_client_push_sparse_rpc");
        closure->add_timer(rpc_timer);

        std::vector<std::future<int>> merge_status(request_call_num);
        for (size_t shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
          merge_status[shard_idx] = async_push_sparse_shard_threads.enqueue(
              std::bind(&BrpcPsClient::PushSparseAsyncShardPush,
                        this,
                        task_list,
                        request_kv_num,
                        table_id,
                        shard_idx,
                        closure,
                        accessor));
        }
        for (size_t shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
          merge_status[shard_idx].wait();
        }
        merge_status.clear();
        std::vector<std::future<int>>().swap(merge_status);
        _push_sparse_merge_count_map[table_id] = 0;
      } else {  // 未达到阈值 只做多路归并
        std::vector<std::future<int>> merge_status(request_call_num);
        for (size_t shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
          merge_status[shard_idx] = async_push_sparse_shard_threads.enqueue(
              std::bind(&BrpcPsClient::PushSparseAsyncShardMerge,
                        this,
                        task_list,
                        request_kv_num,
                        table_id,
                        shard_idx,
                        accessor));
        }
        for (size_t shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
          merge_status[shard_idx].wait();
        }

        // merge到task_list[0]
        auto async_task = new SparseAsyncTask(*(task_list[0].get()));

        task_queue->Put(std::move(async_task));
        --_async_call_num;
        merge_status.clear();
        std::vector<std::future<int>>().swap(merge_status);
      }
    }
    auto wait_ms = FLAGS_pserver_async_push_sparse_interval_ms -
                   (butil::gettimeofday_ms() - async_start_time_ms);
    if (wait_ms > 0) {
      usleep(wait_ms * 1000);
    }
  }
}

void sparse_local_merge(ValueAccessor *accessor,
                        float *merge_data,
                        const float *another_data) {
  size_t col_num = accessor->GetAccessorInfo().update_dim;
  std::vector<float *> merge_data_shell(col_num);
  std::vector<const float *> another_data_shell(col_num);
  for (size_t i = 0; i < col_num; ++i) {
    merge_data_shell[i] = merge_data + i;
    another_data_shell[i] = another_data + i;
  }
  accessor->Merge(merge_data_shell.data(), another_data_shell.data(), 1);
}

int BrpcPsClient::PushSparseAsyncShardMerge(
    std::vector<std::shared_ptr<SparseAsyncTask>> &task_list,
    std::vector<int> &request_kv_num,
    int table_id,
    int shard_idx,
    ValueAccessor *accessor) {
  size_t merged_kv_count = 0;
  uint32_t value_size = accessor->GetAccessorInfo().update_size;

  thread_local std::vector<std::pair<uint64_t, const float *>> sorted_kv_list;
  sorted_kv_list.clear();
  for (size_t i = 1; i < task_list.size(); ++i) {
    size_t kv_num = task_list[i]->data()->shared_data[shard_idx].kv_num;
    auto &key_list = task_list[i]->data()->shared_data[shard_idx].key_list;
    auto &value_list = task_list[i]->data()->shared_data[shard_idx].value_list;

    for (size_t j = 0; j < kv_num; ++j) {
      if (value_list[j].size() < value_size) {
        LOG(WARNING) << "value_list[" << j << "]: " << value_list[j].c_str()
                     << "is invalid.";
        continue;
      }
      char *task_data_ptr = const_cast<char *>(value_list[j].data());
      sorted_kv_list.push_back(
          {key_list[j], reinterpret_cast<float *>(task_data_ptr)});
    }
  }

  // 按key排序&去重
  std::sort(sorted_kv_list.begin(),
            sorted_kv_list.end(),
            [](const std::pair<uint64_t, const float *> &k1,
               const std::pair<uint64_t, const float *> &k2) {
              return k1.first < k2.first;
            });

  auto &async_task = task_list[0];
  size_t sorted_kv_size = sorted_kv_list.size();
  auto &shard_kv_data = async_task->data()->shared_data[shard_idx];
  shard_kv_data.key_list.resize(sorted_kv_size);
  shard_kv_data.value_list.resize(sorted_kv_size);

  // 将去重后数据写入分shard包
  if (sorted_kv_size == 0) {
    shard_kv_data.kv_num = 0;
    return 0;
  } else if (sorted_kv_size == 1) {
    shard_kv_data.kv_num = 1;
    shard_kv_data.key_list[0] = sorted_kv_list[0].first;
    shard_kv_data.value_list[0].assign((const char *)(sorted_kv_list[0].second),
                                       value_size);
    return 0;
  }

  // 去重 本地merge
  uint64_t last_key = sorted_kv_list[0].first;
  const float *last_value_data = sorted_kv_list[0].second;
  float *last_merge_data = nullptr;
  std::shared_ptr<char> merger_buffer(new char[value_size],
                                      array_deleter<char>());
  for (size_t kv_idx = 1; kv_idx < sorted_kv_size; ++kv_idx) {
    while (kv_idx < sorted_kv_size &&
           last_key == sorted_kv_list[kv_idx].first) {
      if (last_merge_data == nullptr) {
        last_merge_data = reinterpret_cast<float *>(merger_buffer.get());
        memcpy(last_merge_data, last_value_data, value_size);
      }
      sparse_local_merge(
          accessor, last_merge_data, sorted_kv_list[kv_idx].second);
      ++kv_idx;
    }
    if (last_merge_data != nullptr) {
      shard_kv_data.value_list[merged_kv_count].assign(
          (const char *)last_merge_data, value_size);
      last_merge_data = nullptr;
    } else {
      shard_kv_data.value_list[merged_kv_count].assign(
          (const char *)sorted_kv_list[kv_idx - 1].second, value_size);
    }
    shard_kv_data.key_list[merged_kv_count++] = last_key;
    if (kv_idx < sorted_kv_size) {
      last_key = sorted_kv_list[kv_idx].first;
      last_value_data = sorted_kv_list[kv_idx].second;
    }
    if (kv_idx == sorted_kv_size - 1) {
      shard_kv_data.value_list[merged_kv_count].assign(
          (const char *)last_value_data, value_size);
      shard_kv_data.key_list[merged_kv_count++] = last_key;
    }
  }
  shard_kv_data.kv_num = merged_kv_count;
  return 0;
}

int BrpcPsClient::PushSparseAsyncShardPush(
    std::vector<std::shared_ptr<SparseAsyncTask>> &task_list,
    std::vector<int> &request_kv_num,
    int table_id,
    int shard_idx,
    DownpourBrpcClosure *closure,
    ValueAccessor *accessor) {
  PushSparseAsyncShardMerge(
      task_list, request_kv_num, table_id, shard_idx, accessor);
  size_t merged_kv_count = task_list[0]->data()->shared_data[shard_idx].kv_num;

  auto &merged_key_list = task_list[0]->data()->shared_data[shard_idx].key_list;
  auto &merged_value_list =
      task_list[0]->data()->shared_data[shard_idx].value_list;

  // 发送RPC请求
  auto *push_request = closure->request(shard_idx);
  push_request->set_cmd_id(PS_PUSH_SPARSE_TABLE);
  push_request->set_table_id(table_id);
  push_request->set_client_id(_client_id);
  push_request->add_params(reinterpret_cast<char *>(&merged_kv_count),
                           sizeof(uint32_t));  // NOLINT
  auto *push_data = push_request->mutable_data();
  int update_size = accessor->GetAccessorInfo().update_size;
  push_data->resize(merged_kv_count * (sizeof(uint64_t) + update_size));
  char *push_data_ptr = const_cast<char *>(push_data->data());
  memcpy(push_data_ptr,
         merged_key_list.data(),
         merged_kv_count * sizeof(uint64_t));
  push_data_ptr += merged_kv_count * sizeof(uint64_t);
  for (size_t i = 0; i < merged_kv_count; ++i) {
    const char *task_data_ptr = merged_value_list[i].data();

    memcpy(push_data_ptr,
           (float *)(task_data_ptr),  // NOLINT
           update_size);
    push_data_ptr += update_size;
  }
  PsService_Stub rpc_stub(GetSparseChannel(shard_idx));
  closure->cntl(shard_idx)->set_request_compress_type(
      (brpc::CompressType)FLAGS_pserver_communicate_compress_type);
  rpc_stub.service(closure->cntl(shard_idx),
                   closure->request(shard_idx),
                   closure->response(shard_idx),
                   closure);
  _push_sparse_merge_count_map[table_id] = 0;
  return 0;
}

std::future<int32_t> BrpcPsClient::PushDense(const Region *regions,
                                             size_t region_num,
                                             size_t table_id) {
  auto *accessor = GetTableAccessor(table_id);
  int fea_dim = accessor->GetAccessorInfo().fea_dim;
  int update_dim = accessor->GetAccessorInfo().update_dim;
  auto push_timer = std::make_shared<CostTimer>("pserver_client_push_dense");
  auto parse_timer =
      std::make_shared<CostTimer>("pserver_client_push_dense_parse");
  int push_dense_async_num = _push_dense_task_queue_map[table_id]->Size();
  while (push_dense_async_num > FLAGS_pserver_max_async_call_num) {
    //    LOG(INFO) << "PushDense Waiting for async_call_num consume,
    //    task_num:"
    //              << push_dense_async_num
    //              << ", max_task_limit:" << FLAGS_pserver_max_async_call_num;
    usleep(5000);  // 5ms
    push_dense_async_num = _push_dense_task_queue_map[table_id]->Size();
  }
  auto push_dense_timer = std::make_shared<CostTimer>("push_dense_put");
  // auto dense_data = _dense_matrix_obj_pool.get();
  auto dense_data = std::make_shared<std::vector<float>>();
  auto async_task = new DenseAsyncTask(dense_data, table_id, push_timer);
  size_t request_call_num = _server_channels.size();
  uint32_t num_per_shard = DenseDimPerShard(fea_dim, request_call_num);
  // 将region数据拷贝到转置矩阵中
  async_task->data()->resize(num_per_shard * request_call_num * update_dim);
  float *data = async_task->data()->data();
  size_t data_size = async_task->data()->size();
  uint32_t pos = 0;
  for (size_t i = 0; i < region_num; ++i) {
    uint32_t data_num = regions[i].size / sizeof(float);
    PADDLE_ENFORCE_LE((pos + data_num),
                      data_size,
                      common::errors::InvalidArgument(
                          "Invalid dense size."
                          "Expect the sum of current position and data number "
                          "to be equal to or smaller than the size."
                          "But received current position = %lu, data number = "
                          "%lu, size = %lu.",
                          pos,
                          data_num,
                          data_size));
    const float *region_data = (const float *)(regions[i].data);
    memcpy(data + pos, region_data, regions[i].size);
    pos += data_num;
  }
  std::future<int> fut = async_task->get_future();
  _push_dense_task_queue_map[table_id]->Put(std::move(async_task));
  return fut;
}

void BrpcPsClient::PushDenseTaskConsume() {
  uint64_t merge_size = FLAGS_pserver_push_dense_merge_limit;
  static bool scale_gradient = FLAGS_pserver_scale_gradient_by_merge;
  ::ThreadPool async_merge_dense_threads(10);
  while (_running) {
    auto async_start_time_ms = butil::gettimeofday_ms();
    for (auto &task_queue_itr : _push_dense_task_queue_map) {
      auto &task_queue = task_queue_itr.second;
      auto queue_size = task_queue->Size();
      if (queue_size == 0) {
        continue;
      }
      if (queue_size <= merge_size && _flushing == false) {
        continue;
      }
      ++_async_call_num;
      DenseAsyncTask *task = nullptr;
      task_queue->Get(task);
      auto *accessor = GetTableAccessor(task->table_id());
      // 设置请求回调
      size_t request_call_num = _server_channels.size();

      DownpourBrpcClosure *closure = new DownpourBrpcClosure(
          request_call_num, [this, request_call_num](void *done) {
            int ret = 0;
            auto *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
            for (size_t i = 0; i < request_call_num; ++i) {
              if (closure->check_response(i, PS_PUSH_DENSE_TABLE) != 0) {
                ret = -1;
                break;
              }
            }
            closure->set_promise_value(ret);
            --_async_call_num;
          });

      auto &total_send_data_vec = *(task->data());
      float *total_send_data =
          reinterpret_cast<float *>(total_send_data_vec.data());
      size_t total_send_data_size = total_send_data_vec.size();
      {
        CostTimer merge_timer("pserver_client_push_dense_merge");
        uint32_t merge_count = 0;
        std::vector<std::future<int>> merge_status(merge_size);
        while (!task_queue->Empty() && merge_count < merge_size) {
          auto *async_task = new DenseAsyncTask();
          task_queue->Get(async_task);
          closure->add_timer(async_task->timer());
          closure->add_promise(async_task->promise());
          merge_status[merge_count] =
              async_merge_dense_threads.enqueue([accessor,
                                                 &total_send_data,
                                                 total_send_data_size,
                                                 async_task]() -> int {
                auto &tmp_task_vec = *(async_task->data());
                const float *merge_data = tmp_task_vec.data();
                accessor->Merge(
                    &total_send_data, &merge_data, total_send_data_size);
#ifndef __GNUC__
#pragma optimize("", off)
#endif
                delete async_task;
#ifndef __GNUC__
#pragma optimize("", on)
#endif
                return 0;
              });
          ++merge_count;
        }
        for (size_t i = 0; i < merge_count; ++i) {
          merge_status[i].wait();
        }

        VLOG(3) << "BrpcPsClient::PushDenseTaskConsume before merge "
                   "total_send_data[0]"
                << total_send_data[0] << " total_send_data[-2]"
                << total_send_data[total_send_data_size - 2]
                << total_send_data[0] << " total_send_data[-1]"
                << total_send_data[total_send_data_size - 1];

        if (scale_gradient && merge_count > 1) {
          Eigen::Map<Eigen::MatrixXf> mat(
              total_send_data, 1, total_send_data_size);
          mat *= (1.0 / (merge_count + 1));
        }

        VLOG(3) << "BrpcPsClient::PushDenseTaskConsume after merge "
                   "total_send_data[0]"
                << total_send_data[0] << " total_send_data[-2]"
                << total_send_data[total_send_data_size - 2]
                << " total_send_data[-1]"
                << total_send_data[total_send_data_size - 1] << " merge_count "
                << merge_count;
      }
      std::shared_ptr<DenseAsyncTask> task_ptr(task);
      PushDenseRawGradient(
          task_ptr, total_send_data, total_send_data_size, closure);
    }
    auto wait_ms = FLAGS_pserver_async_push_dense_interval_ms -
                   (butil::gettimeofday_ms() - async_start_time_ms);
    if (wait_ms > 0) {
      usleep(wait_ms * 1000);
    }
  }
}

void BrpcPsClient::PushDenseRawGradient(std::shared_ptr<DenseAsyncTask> &task,
                                        float *total_send_data,
                                        size_t total_send_data_size,
                                        DownpourBrpcClosure *closure) {
  auto *accessor = GetTableAccessor(task->table_id());
  size_t request_call_num = _server_channels.size();
  // 将数据拷贝到请求buffer区
  auto timer = std::make_shared<CostTimer>("pserver_client_push_dense_rpc");
  closure->add_timer(timer);
  uint32_t num_per_shard =
      DenseDimPerShard(accessor->GetAccessorInfo().fea_dim, request_call_num);
  auto send_timer =
      std::make_shared<CostTimer>("pserver_client_push_dense_send");
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PUSH_DENSE_TABLE);
    closure->request(i)->set_table_id(task->table_id());
    closure->request(i)->set_client_id(_client_id);
    auto *push_data = closure->request(i)->mutable_data();
    push_data->clear();
    push_data->resize(sizeof(uint32_t) + num_per_shard * sizeof(float));
    char *push_data_ptr = const_cast<char *>(push_data->data());
    memcpy(push_data_ptr, &num_per_shard, sizeof(uint32_t));
    memcpy(push_data_ptr + sizeof(uint32_t),
           total_send_data + i * num_per_shard,
           num_per_shard * sizeof(float));
    closure->cntl(i)->set_request_compress_type(
        (brpc::CompressType)FLAGS_pserver_communicate_compress_type);
    PsService_Stub rpc_stub(GetDenseChannel(i));
    rpc_stub.service(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
  }
}

}  // namespace paddle::distributed
