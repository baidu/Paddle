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

#include "paddle/fluid/distributed/ps/service/heter_server.h"

#include "paddle/utils/string/split.h"

namespace paddle::distributed {
// PD_DEFINE_string(cert_path, "./cert.pem", "cert.pem path");
// PD_DEFINE_string(key_path, "./key.pem", "key.pem path");
std::shared_ptr<HeterServer> HeterServer::s_instance_ = nullptr;
std::mutex HeterServer::mtx_;

void HeterServer::RegisterServiceHandler(std::string message_name,
                                         HeterServiceHandler func) {
  service_.RegisterServiceHandler(message_name, func);
}

void HeterServer::StartHeterService(bool need_encrypt) {
  server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE);
  brpc::ServerOptions options;
  if (need_encrypt) {
    options.mutable_ssl_options()->default_cert.certificate = "/cert.pem";
    options.mutable_ssl_options()->default_cert.private_key = "/key.pem";
  }
  if (server_.Start(endpoint_.c_str(), &options) != 0) {
    VLOG(0) << "HeterServer start fail. Try again.";
    auto ip_port = ::paddle::string::Split(endpoint_, ':');
    std::string ip = ip_port[0];
    int port = std::stoi(ip_port[1]);
    std::string int_ip_port = GetIntTypeEndpoint(ip, port);
    if (server_.Start(endpoint_.c_str(), &options) != 0) {
      LOG(ERROR) << "HeterServer start failed, ip_port= " << int_ip_port;
    }
  } else {
    VLOG(0) << "heter server start success! listen on " << endpoint_;
  }

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    stopped_ = false;
    ready_ = 1;
  }
  condition_ready_.notify_all();
  VLOG(4) << "stopped: " << stopped_ << ", ready_: " << ready_;
  std::unique_lock<std::mutex> running_lock(mutex_);
  cv_.wait(running_lock, [&] {
    VLOG(4) << "Heter Server is Stop? " << stopped_;
    return stopped_;
  });
  VLOG(4) << "start service done";
}

void HeterServer::StartHeterInterService(bool need_encrypt) {
  server_inter_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE);
  brpc::ServerOptions options;
  if (need_encrypt) {
    options.mutable_ssl_options()->default_cert.certificate = "/cert.pem";
    options.mutable_ssl_options()->default_cert.private_key = "/key.pem";
  }
  if (server_inter_.Start(endpoint_inter_.c_str(), &options) != 0) {
    VLOG(4) << "switch inter server start fail. Try again.";
    auto ip_port = ::paddle::string::Split(endpoint_inter_, ':');
    std::string ip = ip_port[0];
    int port = std::stoi(ip_port[1]);
    std::string int_ip_port = GetIntTypeEndpoint(ip, port);
    if (server_inter_.Start(endpoint_inter_.c_str(), &options) != 0) {
      LOG(ERROR) << "switch inter server start failed, ip_port= "
                 << int_ip_port;
    }
  } else {
    VLOG(4) << "switch inter server server start success! listen on "
            << endpoint_inter_;
  }
  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    stopped_ = false;
    ready_ = 1;
  }
  condition_ready_.notify_all();
  VLOG(4) << "stopped: " << stopped_ << ", ready_: " << ready_;
  std::unique_lock<std::mutex> running_lock(mutex_);
  cv_.wait(running_lock, [&] {
    VLOG(4) << "Heter Server is Stop? " << stopped_;
    return stopped_;
  });
  VLOG(4) << "start service done";
}

void HeterServer::SetFanIn(const int& fan_in) { service_.SetFanIn(fan_in); }

void HeterServer::WaitServerReady() {
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
}

int SendAndRecvVariableHandler::SaveInSwitchWithShard(
    const MultiVarMsg* request,
    PsResponseMessage* response,
    brpc::Controller* cntl) {
  VLOG(4) << "entering SaveInSwitchWithShard";
  int32_t group_id = request->group_id();
  if (group_id >= FLAGS_heter_world_size) {
    LOG(ERROR) << "group id exceed maximum";
  }
  auto& local_shard = _local_shards[group_id];
  auto& request_io_buffer = cntl->request_attachment();
  butil::IOBufBytesIterator io_buffer_itr(request_io_buffer);
  for (int idx = 0; idx < request->send_var_names_size(); idx++) {
    const auto& var_name = request->send_var_names(idx);
    const auto& var_size = request->vars_len(idx);
    WaitForVarsConsumed(group_id, var_name);
    std::unique_lock<std::mutex> lk(scope_mutex_);
    auto& value = local_shard[var_name];
    value.resize(var_size);
    io_buffer_itr.copy_and_forward(reinterpret_cast<void*>(value.data()),
                                   var_size);
    vars_ready_flag[group_id][var_name] = 1;
    VLOG(4) << "saved var_name: " << var_name << "is saved ready!";
  }
  VLOG(4) << "SaveInSwitchWithShard success";
  return 0;
}

int SendAndRecvVariableHandler::QueryInSwitchWithShard(
    const MultiVarMsg* request, MultiVarMsg* response, brpc::Controller* cntl) {
  VLOG(4) << "entering QueryInSwitchWithShard";
  int32_t group_id = request->group_id();
  VLOG(4) << "group id: " << group_id;
  auto& local_shard = _local_shards[group_id];
  auto& response_io_buffer = cntl->response_attachment();
  auto req_var_nums = request->recv_var_names_size();
  std::vector<std::string> req_var_names(req_var_nums);
  for (int var_idx = 0; var_idx < req_var_nums; ++var_idx) {
    req_var_names[var_idx] = request->recv_var_names(var_idx);
  }
  auto msg_name = request->message_name();
  response->set_message_name(msg_name);
  for (auto& req_var_name : req_var_names) {
    VLOG(4) << "req var name: " << req_var_name;
    response->add_send_var_names(req_var_name);
    WaitForVarsProduced(group_id, req_var_name);
    std::unique_lock<std::mutex> lk(scope_mutex_);
    auto itr = local_shard.find(req_var_name);
    auto& value = itr.value();
    response_io_buffer.append(value.data(), value.size());
    value.resize(0);  // 清空内存
    vars_ready_flag[group_id][req_var_name] = 0;
    VLOG(4) << "query var_name: " << req_var_name << "is consumed ready!";
  }
  VLOG(4) << "heter server QueryInSwitchWithShard done";
  return 0;
}

int SendAndRecvVariableHandler::SaveInSwitchWithScope(
    const MultiVarMsg* request,
    PsResponseMessage* response,
    brpc::Controller* cntl) {
  VLOG(4) << "entering SaveInSwitchWithScope";
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  phi::CPUPlace cpu_place;
  auto& cpu_dev_ctx = *pool.Get(cpu_place);
  auto message_name = request->message_name();
  VLOG(4) << "message_name in heter server: " << message_name;

  auto send_var_nums = request->send_var_names_size();
  std::vector<std::string> send_var_names(send_var_nums);
  for (int idx = 0; idx < send_var_nums; idx++) {
    send_var_names[idx] = request->var_messages(idx).varname();
  }
  std::unique_lock<std::mutex> lk(scope_mutex_);
  auto local_scope = local_scope_ptr.get();
  if (!local_scope) {
    LOG(ERROR) << "local_scope_ptr is null in SaveInSwitchWithScope";
  }
  for (auto var_name : send_var_names) {
    auto* var_exist_ptr = local_scope->FindVar(var_name);
    if (!var_exist_ptr) {
      VLOG(4) << "not find var: " << var_name << " in local_scope";
    }
    WaitForVarsConsumed(0, var_name);
  }
  auto& request_io_buffer = cntl->request_attachment();
  distributed::DeserializeFromMultiVarMsgAndIOBuf(
      *request, &request_io_buffer, cpu_dev_ctx, local_scope);
  lk.unlock();
  for (auto var_name : send_var_names) {
    std::unique_lock<std::mutex> lk(scope_mutex_);
    vars_ready_flag[0][var_name] = 1;
  }
  VLOG(4) << "SaveInSwitchWithScope success";
  return 0;
}

int SendAndRecvVariableHandler::QueryInSwitchWithScope(
    const MultiVarMsg* request, MultiVarMsg* response, brpc::Controller* cntl) {
  VLOG(4) << "entering QueryInSwitchWithScope";
  auto local_scope = local_scope_ptr.get();
  if (!local_scope) {
    LOG(INFO) << "local_scope is null";
  }
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  phi::CPUPlace cpu_place;
  auto& cpu_dev_ctx = *pool.Get(cpu_place);

  // get req message_name & req_var_names
  auto msg_name = request->message_name();
  auto req_var_nums = request->recv_var_names_size();
  std::vector<std::string> req_var_names(req_var_nums);
  for (int var_idx = 0; var_idx < req_var_nums; ++var_idx) {
    req_var_names[var_idx] = request->recv_var_names(var_idx);
  }
  auto& response_io_buffer = cntl->response_attachment();

  // 1. fill message_name(string)
  response->set_message_name(msg_name);

  // 2. fill var_names(string)
  for (auto& req_var_name : req_var_names) {
    response->add_send_var_names(req_var_name);
  }

  // 3. fill var_messages(VarMessage)
  for (auto& req_var_name : req_var_names) {
    WaitForVarsProduced(0, req_var_name);
    auto* send_var_msg = response->add_var_messages();
    send_var_msg->set_varname(req_var_name);

    framework::Variable* var_ptr;
    var_ptr = local_scope->FindVar(req_var_name);
    if (!var_ptr) {
      LOG(INFO) << "local_scope not find var: " << req_var_name;
    }
    butil::IOBuf temp_iobuf;
    if (var_ptr->IsType<phi::DenseTensor>()) {
      SerializeDenseTensor(var_ptr, cpu_dev_ctx, send_var_msg, &temp_iobuf);
    } else if (var_ptr->IsType<phi::SelectedRows>()) {
      SerializeSelectedRows(var_ptr, cpu_dev_ctx, send_var_msg, &temp_iobuf);
    }
    response_io_buffer.append(temp_iobuf);
  }
  for (auto& req_var_name : req_var_names) {
    std::unique_lock<std::mutex> lk(scope_mutex_);
    vars_ready_flag[0][req_var_name] = 0;
  }
  VLOG(4) << "heter server QueryInSwitchWithScope done";
  return 0;
}
}  // namespace paddle::distributed
