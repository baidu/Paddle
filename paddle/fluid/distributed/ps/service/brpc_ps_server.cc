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

#include "paddle/fluid/distributed/ps/service/brpc_ps_server.h"

#include <thread>  // NOLINT

#include "butil/object_pool.h"
#include "paddle/fluid/distributed/common/cost_timer.h"
#include "paddle/fluid/distributed/ps/table/depends/sparse_utils.h"
#include "paddle/fluid/distributed/ps/table/table.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/phi/core/platform/profiler.h"

namespace google::protobuf {
class Closure;
class RpcController;
}  // namespace google::protobuf

PD_DEFINE_int32(pserver_timeout_ms_s2s,
                10000,
                "pserver request server timeout_ms");
PD_DEFINE_int32(pserver_connect_timeout_ms_s2s,
                10000,
                "pserver connect server timeout_ms");
PD_DEFINE_string(pserver_connection_type_s2s,
                 "pooled",
                 "pserver connection_type[pooled:single]");

namespace paddle::distributed {

int32_t BrpcPsServer::Initialize() {
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

uint64_t BrpcPsServer::Start(const std::string &ip, uint32_t port) {
  std::unique_lock<std::mutex> lock(mutex_);

  std::string ip_port = ip + ":" + std::to_string(port);
  VLOG(0) << "running server with rank id: " << _rank
          << ", endpoint: " << ip_port;
  brpc::ServerOptions options;

  int num_threads = std::thread::hardware_concurrency();
  auto trainers = _environment->GetTrainers();
  options.num_threads = trainers > num_threads ? trainers : num_threads;

  if (_server.Start(ip_port.c_str(), &options) != 0) {
    VLOG(0) << "BrpcPsServer start failed, ip_port= " << ip_port
            << " , Try Again.";

    std::string int_ip_port = GetIntTypeEndpoint(ip, port);

    if (_server.Start(int_ip_port.c_str(), &options) != 0) {
      LOG(ERROR) << "BrpcPsServer start failed, ip_port= " << int_ip_port;
      return 0;
    }
  }

  _environment->RegisterPsServer(ip, port, _rank);
  cv_.wait(lock, [&] { return stopped_; });

  PSHost host;
  host.ip = ip;
  host.port = port;
  host.rank = _rank;
  return host.rank;
}

int32_t BrpcPsServer::StartS2S() {
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.timeout_ms = FLAGS_pserver_timeout_ms_s2s;
  options.connection_type = FLAGS_pserver_connection_type_s2s;
  options.connect_timeout_ms = FLAGS_pserver_connect_timeout_ms_s2s;
  options.max_retry = 3;

  std::vector<PSHost> pserver_list = _environment->GetPsServers();
  _pserver_channels.resize(pserver_list.size());
  VLOG(2) << "pserver start s2s server_list size: " << _pserver_channels.size();

  std::ostringstream os;
  std::string server_ip_port;

  for (size_t i = 0; i < pserver_list.size(); ++i) {
    server_ip_port.assign(pserver_list[i].ip.c_str());
    server_ip_port.append(":");
    server_ip_port.append(std::to_string(pserver_list[i].port));
    _pserver_channels[i].reset(new brpc::Channel());
    if (_pserver_channels[i]->Init(server_ip_port.c_str(), "", &options) != 0) {
      LOG(ERROR) << "pserver connect to pserver:" << server_ip_port
                 << " Failed!";
    }
    os << server_ip_port << ",";
  }
  LOG(INFO) << "pserver connect success: " << os.str();
  return 0;
}

std::future<int32_t> BrpcPsServer::SendPServer2PServerMsg(
    int msg_type, int to_pserver_id, const std::string &msg) {
  auto promise = std::make_shared<std::promise<int32_t>>();
  std::future<int> fut = promise->get_future();
  if (static_cast<size_t>(to_pserver_id) >= _pserver_channels.size()) {
    std::stringstream ss;
    ss << "to_pserver_id is out of range pservers, which size is "
       << _pserver_channels.size();
    PADDLE_THROW(common::errors::Fatal(ss.str()));
    promise->set_value(-1);
    return fut;
  }
  auto *closure = new DownpourPServerBrpcClosure(1, [msg_type](void *done) {
    auto *closure = reinterpret_cast<DownpourPServerBrpcClosure *>(done);
    int32_t ret = closure->check_response(0, msg_type + 1000);
    closure->set_promise_value(ret);
  });

  closure->add_promise(promise);
  closure->request(0)->set_cmd_id(101);
  closure->request(0)->set_client_id(_rank);
  closure->request(0)->set_table_id(0);
  closure->request(0)->set_data(msg);
  PsService_Stub rpc_stub(_pserver_channels[to_pserver_id].get());
  rpc_stub.service(
      closure->cntl(0), closure->request(0), closure->response(0), closure);
  return fut;
}

int32_t BrpcPsServer::ReceiveFromPServer(int msg_type,
                                         int pserver_id,
                                         const std::string &msg) {
  if (msg.length() == 0) {
    LOG(WARNING) << "SERVER>>RESPONSE>>msg = 0 Finish S2S Response";
    return 0;
  }
  ::paddle::framework::BinaryArchive ar;
  ar.SetReadBuffer(const_cast<char *>(msg.c_str()), msg.length(), nullptr);
  if (ar.Cursor() == ar.Finish()) {
    LOG(WARNING) << "SERVER>>RESPONSE ar = 0>> Finish S2S Response";
    return 0;
  }
  std::vector<std::pair<uint64_t, std::string>> data;
  while (ar.Cursor() < ar.Finish()) {
    data.push_back(ar.Get<std::pair<uint64_t, std::string>>());
  }
  PADDLE_ENFORCE_EQ(ar.Cursor(),
                    ar.Finish(),
                    common::errors::InvalidArgument(
                        "Expected 'ar.Cursor()' to be equal to 'ar.Finish()', "
                        "but found they are not equal."));
  this->_shuffled_ins->Write(std::move(data));
  return 0;
}

int32_t BrpcPsServer::Port() { return _server.listen_address().port; }

int32_t BrpcPsService::Initialize() {
  _is_initialize_shard_info = false;
  _service_handler_map[PS_STOP_SERVER] = &BrpcPsService::StopServer;
  _service_handler_map[PS_PULL_DENSE_TABLE] = &BrpcPsService::PullDense;
  _service_handler_map[PS_PUSH_DENSE_TABLE] = &BrpcPsService::PushDense;
  _service_handler_map[PS_PULL_SPARSE_TABLE] = &BrpcPsService::PullSparse;
  _service_handler_map[PS_PUSH_SPARSE_TABLE] = &BrpcPsService::PushSparse;
  _service_handler_map[PS_SAVE_ONE_TABLE] = &BrpcPsService::SaveOneTable;
  _service_handler_map[PS_SAVE_ALL_TABLE] = &BrpcPsService::SaveAllTable;
  _service_handler_map[PS_SHRINK_TABLE] = &BrpcPsService::ShrinkTable;
  _service_handler_map[PS_LOAD_ONE_TABLE] = &BrpcPsService::LoadOneTable;
  _service_handler_map[PS_LOAD_ALL_TABLE] = &BrpcPsService::LoadAllTable;
  _service_handler_map[PS_CLEAR_ONE_TABLE] = &BrpcPsService::ClearOneTable;
  _service_handler_map[PS_CLEAR_ALL_TABLE] = &BrpcPsService::ClearAllTable;
  _service_handler_map[PS_PUSH_DENSE_PARAM] = &BrpcPsService::PushDenseParam;
  _service_handler_map[PS_PRINT_TABLE_STAT] = &BrpcPsService::PrintTableStat;
  _service_handler_map[PS_PULL_GEO_PARAM] = &BrpcPsService::PullGeoParam;
  _service_handler_map[PS_PUSH_SPARSE_PARAM] = &BrpcPsService::PushSparseParam;
  _service_handler_map[PS_BARRIER] = &BrpcPsService::Barrier;
  _service_handler_map[PS_START_PROFILER] = &BrpcPsService::StartProfiler;
  _service_handler_map[PS_STOP_PROFILER] = &BrpcPsService::StopProfiler;
  _service_handler_map[PS_PUSH_GLOBAL_STEP] = &BrpcPsService::PushGlobalStep;
  // for save cache
  _service_handler_map[PS_SAVE_ONE_CACHE_TABLE] =
      &BrpcPsService::SaveCacheTable;
  _service_handler_map[PS_GET_CACHE_THRESHOLD] =
      &BrpcPsService::GetCacheThreshold;
  _service_handler_map[PS_CACHE_SHUFFLE] = &BrpcPsService::CacheShuffle;

  _service_handler_map[PS_REVERT] = &BrpcPsService::Revert;
  _service_handler_map[PS_CHECK_SAVE_PRE_PATCH_DONE] =
      &BrpcPsService::CheckSavePrePatchDone;

  auto &profiler = CostProfiler::instance();
  profiler.register_profiler("pserver_server_pull_dense");
  profiler.register_profiler("pserver_server_push_dense");
  profiler.register_profiler("pserver_server_pull_sparse");
  profiler.register_profiler("pserver_server_push_sparse");

  // shard初始化,server启动后才可从env获取到server_list的shard信息
  InitializeShardInfo();

  return 0;
}

#define CHECK_TABLE_EXIST(table, request, response)        \
  if (table == NULL) {                                     \
    std::string err_msg("table not found with table_id:"); \
    err_msg.append(std::to_string(request.table_id()));    \
    set_response_code(response, -1, err_msg.c_str());      \
    return -1;                                             \
  }

int32_t BrpcPsService::InitializeShardInfo() {
  if (!_is_initialize_shard_info) {
    std::lock_guard<std::mutex> guard(_initialize_shard_mutex);
    if (_is_initialize_shard_info) {
      return 0;
    }
    size_t shard_num = _server->Environment()->GetPsServers().size();
    auto &table_map = *(_server->GetTable());
    for (auto itr : table_map) {
      itr.second->SetShard(_rank, shard_num);
    }
    _is_initialize_shard_info = true;
  }
  return 0;
}

void BrpcPsService::service(google::protobuf::RpcController *cntl_base,
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

  if (request->cmd_id() < 100) {
    auto itr = _service_handler_map.find(request->cmd_id());
    if (itr == _service_handler_map.end()) {
      std::string err_msg(
          "undefined cmd_id, should match PsCmdID in ps.proto, cmd_id:");
      err_msg.append(std::to_string(request->cmd_id()));
      set_response_code(*response, -1, err_msg.c_str());
      return;
    }
    serviceHandlerFunc handler_func = itr->second;
    int service_ret = (this->*handler_func)(table, *request, *response, cntl);
    if (service_ret != 0) {
      response->set_err_code(service_ret);
      response->set_err_msg("server internal error");
    }
  } else {
    int service_ret = _server->HandlePServer2PServerMsg(
        request->cmd_id(), request->client_id(), request->data());
    if (service_ret != 0) {
      response->set_err_code(-1);
      response->set_err_msg("handle_pserver2pserver_msg failed");
    }
  }
}

int32_t BrpcPsService::PullDense(Table *table,
                                 const PsRequestMessage &request,
                                 PsResponseMessage &response,
                                 brpc::Controller *cntl) {
  phi::RecordEvent record_event(
      "PsService->PullDense", phi::TracerEventType::Communication, 1);
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 1) {
    set_response_code(
        response,
        -1,
        "PsRequestMessage.datas is required at least 1 for num of dense");
    return 0;
  }
  CostTimer timer("pserver_server_pull_dense");
  uint32_t num = *(const uint32_t *)request.params(0).c_str();

  auto res_data = butil::get_object<std::vector<float>>();
  res_data->resize(num *
                   table->GetValueAccessor()->GetAccessorInfo().select_size /
                   sizeof(float));

  TableContext table_context;
  table_context.value_type = Dense;
  table_context.pull_context.values = res_data->data();
  table_context.num = num;
  table->Pull(table_context);

  cntl->response_attachment().append(reinterpret_cast<char *>(res_data->data()),
                                     res_data->size() * sizeof(float));
  butil::return_object(res_data);

  return 0;
}

int32_t BrpcPsService::PushDenseParam(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  phi::RecordEvent record_event(
      "PsService->PushDenseParam", phi::TracerEventType::Communication, 1);
  CHECK_TABLE_EXIST(table, request, response)
  thread_local std::string push_buffer;
  auto &req_io_buffer = cntl->request_attachment();
  auto req_buffer_size = req_io_buffer.size();
  if (req_buffer_size < 1) {
    set_response_code(response, -1, "req attachment is empty");
    return 0;
  }
  push_buffer.resize(0);
  push_buffer.reserve(req_buffer_size);
  const char *data = (const char *)cntl->request_attachment().fetch(
      const_cast<char *>(push_buffer.data()), req_buffer_size);

  uint32_t num = *(const uint32_t *)data;

  const float *values = (const float *)(data + sizeof(uint32_t));
  TableContext table_context;
  table_context.value_type = Dense;
  table_context.push_context.values = values;
  table_context.push_context.is_param = true;
  table_context.num = num;

  if (table->Push(table_context) != 0) {
    set_response_code(response, -1, "PushDenseParam failed");
  }
  return 0;
}

int32_t BrpcPsService::PushDense(Table *table,
                                 const PsRequestMessage &request,
                                 PsResponseMessage &response,
                                 brpc::Controller *cntl) {
  phi::RecordEvent record_event(
      "PsService->PushDense", phi::TracerEventType::Communication, 1);
  CHECK_TABLE_EXIST(table, request, response)
  auto req_buffer_size = request.data().size();
  if (req_buffer_size < 1) {
    // set_response_code(response, 0, "push dense data is empty");
    return 0;
  }

  CostTimer timer("pserver_server_push_dense");
  /*
  Push Content:
  |--num--|---valuesData---|
  |--4B---|----------------|
  */
  uint32_t num = *(const uint32_t *)(request.data().data());
  TableContext table_context;
  table_context.value_type = Dense;
  table_context.push_context.values =
      (const float *)(request.data().data() + sizeof(uint32_t));
  table_context.num = num;
  // const float *values = (const float *)(request.data().data() +
  // sizeof(uint32_t));
  if (table->Push(table_context) != 0) {
    // if (table->PushDense(values, num) != 0) {
    set_response_code(response, -1, "PushDense failed");
  }

  return 0;
}

int32_t BrpcPsService::Barrier(Table *table,
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

int32_t BrpcPsService::PushSparseParam(Table *table,
                                       const PsRequestMessage &request,
                                       PsResponseMessage &response,
                                       brpc::Controller *cntl) {
  phi::RecordEvent record_event(
      "PsService->PushSparseParam", phi::TracerEventType::Communication, 1);
  CHECK_TABLE_EXIST(table, request, response)
  auto &push_data = request.data();
  if (push_data.empty()) {
    // set_response_code(response, 0, "push sparse data is empty");
    return 0;
  }
  if (request.params_size() < 1) {
    set_response_code(response,
                      -1,
                      "PsRequestMessage.params is required at "
                      "least 1 for num of sparse_key");
    return 0;
  }
  const uint32_t num =
      *(reinterpret_cast<const uint32_t *>(request.params(0).c_str()));
  /*
  Push Content:
  |---keysData---|---valuesData---|
  |---8*{num}B---|----------------|
  */
  const uint64_t *keys = (const uint64_t *)push_data.data();
  const float *values =
      (const float *)(push_data.data() + sizeof(uint64_t) * num);

  TableContext table_context;
  table_context.value_type = Sparse;
  table_context.push_context.keys = keys;
  table_context.push_context.values = values;
  table_context.push_context.is_param = true;
  table_context.num = num;
  //  if (table->PushSparseParam(keys, values, num) != 0) {
  if (table->Push(table_context) != 0) {
    set_response_code(response, -1, "PushSparseParam error");
  }
  return 0;
}

int32_t BrpcPsService::PullGeoParam(Table *table,
                                    const PsRequestMessage &request,
                                    PsResponseMessage &response,
                                    brpc::Controller *cntl) {
  phi::RecordEvent record_event(
      "PsService->pull_geo_param", phi::TracerEventType::Communication, 1);
  CHECK_TABLE_EXIST(table, request, response)
  thread_local std::string push_sparse_request_buffer;

  auto trainer_id = request.client_id();

  std::vector<float> values;
  std::vector<uint64_t> ids;

  TableContext table_context;
  table_context.value_type = Sparse;
  table_context.pull_context.geo_pull_keys = &ids;
  table_context.pull_context.geo_pull_values = &values;
  table_context.trainer_id = trainer_id;
  table->Pull(table_context);
  //  table->PullGeoParam(trainer_id, &values, &ids);

  uint32_t num = ids.size();
  cntl->response_attachment().append(reinterpret_cast<char *>(&num),
                                     sizeof(uint32_t));
  cntl->response_attachment().append(reinterpret_cast<char *>(ids.data()),
                                     ids.size() * sizeof(uint64_t));
  cntl->response_attachment().append(reinterpret_cast<char *>(values.data()),
                                     values.size() * sizeof(float));
  return 0;
}

int32_t BrpcPsService::PullSparse(Table *table,
                                  const PsRequestMessage &request,
                                  PsResponseMessage &response,
                                  brpc::Controller *cntl) {
  phi::RecordEvent record_event(
      "PsService->PullSparse", phi::TracerEventType::Communication, 1);
  CHECK_TABLE_EXIST(table, request, response)

  auto &req_io_buffer = cntl->request_attachment();
  auto req_buffer_size = req_io_buffer.size();

  if (req_buffer_size < 1) {
    set_response_code(response, -1, "req attachment is empty");
    return 0;
  }

  if (request.params_size() < 1) {
    set_response_code(response,
                      -1,
                      "PsRequestMessage.params is required at "
                      "least 1 for num of sparse_key");
    return 0;
  }

  CostTimer timer("pserver_server_pull_sparse");
  const uint32_t num =
      *(reinterpret_cast<const uint32_t *>(request.params(0).c_str()));
  auto dim = table->GetValueAccessor()->GetAccessorInfo().select_dim;

  thread_local std::string req_buffer;
  req_buffer.reserve(req_buffer_size);

  const void *data = cntl->request_attachment().fetch(
      const_cast<char *>(req_buffer.data()), req_buffer_size);

  auto value = PullSparseValue(num, dim);

  value.DeserializeFromBytes(const_cast<void *>(data));

  auto res_data = butil::get_object<std::vector<float>>();
  res_data->resize(num * dim);
  TableContext table_context;
  table_context.value_type = Sparse;
  table_context.pull_context.pull_value = value;
  table_context.pull_context.values = res_data->data();
  table->Pull(table_context);
  // table->PullSparse(res_data->data(), value);

  cntl->response_attachment().append(reinterpret_cast<char *>(res_data->data()),
                                     res_data->size() * sizeof(float));
  butil::return_object(res_data);
  return 0;
}

int32_t BrpcPsService::PushSparse(Table *table,
                                  const PsRequestMessage &request,
                                  PsResponseMessage &response,
                                  brpc::Controller *cntl) {
  phi::RecordEvent record_event(
      "PsService->PushSparse", phi::TracerEventType::Communication, 1);
  CHECK_TABLE_EXIST(table, request, response)
  auto &push_data = request.data();
  if (push_data.empty()) {
    // set_response_code(response, 0, "push sparse data is empty");
    return 0;
  }
  if (request.params_size() < 1) {
    set_response_code(response,
                      -1,
                      "PsRequestMessage.params is required at "
                      "least 1 for num of sparse_key");
    return 0;
  }
  CostTimer timer("pserver_server_push_sparse");
  const uint32_t num =
      *(reinterpret_cast<const uint32_t *>(request.params(0).c_str()));
  /*
  Push Content:
  |---keysData---|---valuesData---|
  |---8*{num}B---|----------------|
  */
  TableContext table_context;
  table_context.value_type = Sparse;
  table_context.push_context.keys = (const uint64_t *)push_data.data();
  table_context.push_context.values =
      (const float *)(push_data.data() + sizeof(uint64_t) * num);
  table_context.num = num;
  // const uint64_t *keys = (const uint64_t *)push_data.data();
  // const float *values = (const float *)(push_data.data() + sizeof(uint64_t) *
  // num);
  if (table->Push(table_context) != 0) {
    // if (table->PushSparse(keys, values, num) != 0) {
    set_response_code(response, -1, "PushSparse error");
  }
  return 0;
}

int32_t BrpcPsService::PrintTableStat(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  std::pair<int64_t, int64_t> ret = table->PrintTableStat();
  ::paddle::framework::BinaryArchive ar;
  ar << ret.first << ret.second;
  std::string table_info(ar.Buffer(), ar.Length());
  response.set_data(table_info);

  return 0;
}

int32_t BrpcPsService::LoadOneTable(Table *table,
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

int32_t BrpcPsService::LoadAllTable(Table *table,
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

int32_t BrpcPsService::SaveOneTable(Table *table,
                                    const PsRequestMessage &request,
                                    PsResponseMessage &response,
                                    brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 2) {
    set_response_code(
        response,
        -1,
        "PsRequestMessage.datas is required at least 2, path&mode");
    return -1;
  }
  table->Flush();

  int32_t feasign_size = 0;

  VLOG(3) << "save table " << request.params(0) << " " << request.params(1);
  feasign_size = table->Save(request.params(0), request.params(1));
  if (feasign_size < 0) {
    set_response_code(response, -1, "table save failed");
    return -1;
  }
  return feasign_size;
}

int32_t BrpcPsService::SaveAllTable(Table *table,
                                    const PsRequestMessage &request,
                                    PsResponseMessage &response,
                                    brpc::Controller *cntl) {
  auto &table_map = *(_server->GetTable());
  int32_t feasign_size = 0;

  for (auto &itr : table_map) {
    feasign_size = SaveOneTable(itr.second.get(), request, response, cntl);
    if (feasign_size < 0) {
      LOG(ERROR) << "save table[" << itr.first << "] failed";
      return -1;
    }
  }
  return 0;
}

int32_t BrpcPsService::SaveCacheTable(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 2) {
    set_response_code(
        response,
        -1,
        "PsRequestMessage.datas is required at least 3, path&mode");
    return -1;
  }
  table->Flush();
  int32_t feasign_size = 0;
  // if (_server->_shuffled_ins->size() <= 0) {
  //    LOG(WARNING) << "shuffled ins size <= 0";
  //}
  feasign_size = table->SaveCache(
      request.params(0), request.params(1), _server->_shuffled_ins);
  if (feasign_size < 0) {
    set_response_code(response, -1, "table save failed");
    return -1;
  }
  return feasign_size;
}

int32_t BrpcPsService::CacheShuffle(Table *table,
                                    const PsRequestMessage &request,
                                    PsResponseMessage &response,
                                    brpc::Controller *cntl) {
  // start cache shuffle
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 3) {
    set_response_code(response,
                      -1,
                      "PsRequestMessage.datas is required at least 3, "
                      "path&mode&cache_threshold");
    return -1;
  }
  table->Flush();
  double cache_threshold = std::stod(request.params(2));
  LOG(INFO) << "cache threshold for cache shuffle: " << cache_threshold;
  //    auto shuffled_ins = ::paddle::ps::make_channel<std::pair<uint64_t,
  //    std::string>>();
  //    shuffled_ins->set_block_size(80000);
  _server->StartS2S();
  std::function<std::future<int32_t>(
      int msg_type, int to_pserver_id, const std::string &msg)>
      send_msg_func = [this](int msg_type,
                             int to_pserver_id,
                             const std::string &msg) -> std::future<int32_t> {
    return this->_server->SendPServer2PServerMsg(msg_type, to_pserver_id, msg);
  };

  std::vector<Table *> table_ptrs;
  for (int i = 3; i < request.params_size(); ++i) {
    int table_id = std::stoi(request.params(i));
    Table *table_ptr = _server->GetTable(table_id);
    table_ptrs.push_back(table_ptr);
  }
  if (table_ptrs.empty()) {
    table_ptrs.push_back(table);
  }

  table->CacheShuffle(request.params(0),
                      request.params(1),
                      cache_threshold,
                      send_msg_func,
                      _server->_shuffled_ins,
                      table_ptrs);
  return 0;
}

int32_t BrpcPsService::GetCacheThreshold(Table *table,
                                         const PsRequestMessage &request,
                                         PsResponseMessage &response,
                                         brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  table->Flush();
  double cache_threshold = 0.0;
  cache_threshold = table->GetCacheThreshold();
  if (cache_threshold < 0) {
    LOG(WARNING) << "wrong threshold: " << cache_threshold;
  }
  std::stringstream ss;
  ss << std::setprecision(15) << cache_threshold;
  std::string cache_threshold_str = ss.str();
  response.set_data(cache_threshold_str);
  return 0;
}

int32_t BrpcPsService::Revert(Table *table,
                              const PsRequestMessage &request,
                              PsResponseMessage &response,
                              brpc::Controller *cntl) {
  auto &table_map = *(_server->GetTable());
  for (auto &itr : table_map) {
    itr.second->Flush();
    itr.second->Revert();
  }
  return 0;
}

int32_t BrpcPsService::CheckSavePrePatchDone(Table *table,
                                             const PsRequestMessage &request,
                                             PsResponseMessage &response,
                                             brpc::Controller *cntl) {
  auto &table_map = *(_server->GetTable());
  for (auto &itr : table_map) {
    itr.second->CheckSavePrePatchDone();
  }
  return 0;
}

int32_t BrpcPsService::ShrinkTable(Table *table,
                                   const PsRequestMessage &request,
                                   PsResponseMessage &response,
                                   brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 1) {
    set_response_code(
        response,
        -1,
        "PsRequestMessage.datas is required at least 1, threshold");
    return -1;
  }
  table->Flush();
  if (table->Shrink(request.params(0)) != 0) {
    set_response_code(response, -1, "table shrink failed");
    return -1;
  }
  VLOG(3) << "Pserver Shrink Finished";
  return 0;
}

int32_t BrpcPsService::ClearOneTable(Table *table,
                                     const PsRequestMessage &request,
                                     PsResponseMessage &response,
                                     brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  table->Flush();
  table->Clear();
  return 0;
}

int32_t BrpcPsService::ClearAllTable(Table *table,
                                     const PsRequestMessage &request,
                                     PsResponseMessage &response,
                                     brpc::Controller *cntl) {
  auto &table_map = *(_server->GetTable());
  for (auto &itr : table_map) {
    if (ClearOneTable(itr.second.get(), request, response, cntl) != 0) {
      return -1;
    }
  }
  return 0;
}

int32_t BrpcPsService::StopServer(Table *table,
                                  const PsRequestMessage &request,
                                  PsResponseMessage &response,
                                  brpc::Controller *cntl) {
  auto *p_server = _server;
  std::thread t_stop([p_server]() {
    p_server->Stop();
    VLOG(3) << "Server Stopped";
  });
  t_stop.detach();
  return 0;
}

int32_t BrpcPsService::StopProfiler(Table *table,
                                    const PsRequestMessage &request,
                                    PsResponseMessage &response,
                                    brpc::Controller *cntl) {
  platform::DisableProfiler(platform::EventSortingKey::kDefault,
                            string::Sprintf("server_%s_profile", _rank));
  return 0;
}

int32_t BrpcPsService::StartProfiler(Table *table,
                                     const PsRequestMessage &request,
                                     PsResponseMessage &response,
                                     brpc::Controller *cntl) {
  platform::EnableProfiler(platform::ProfilerState::kCPU);
  return 0;
}

int32_t BrpcPsService::PushGlobalStep(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response);
  auto req_buffer_size = request.data().size();
  if (req_buffer_size < 1) {
    set_response_code(response, 0, "run_program data is empty");
    return 0;
  }
  const int64_t *values =
      (const int64_t *)(request.data().data() + sizeof(uint32_t));
  auto trainer_id = request.client_id();

  TableContext context;
  context.trainer_id = trainer_id;
  context.push_context.push_steps = values;

  //  if (table->PushDense(values, trainer_id) != 0) {
  if (table->Push(context) != 0) {
    set_response_code(response, -1, "run_program failed");
  }

  return 0;
}

}  // namespace paddle::distributed
