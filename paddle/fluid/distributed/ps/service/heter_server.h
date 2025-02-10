/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/common/flags.h"
#include "paddle/common/macros.h"  // for DISABLE_COPY_AND_ASSIGN
#include "paddle/fluid/distributed/ps/service/brpc_utils.h"
#include "paddle/fluid/distributed/ps/service/heter_client.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/ps/table/depends/feature_value.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/phi/core/platform/profiler.h"

namespace google {
namespace protobuf {
class Closure;
class RpcController;
}  // namespace protobuf
}  // namespace google
namespace paddle {
namespace framework {
class Executor;
class ProgramDesc;
class Scope;
}  // namespace framework
}  // namespace paddle
COMMON_DECLARE_double(eager_delete_tensor_gb);
namespace paddle {
namespace distributed {

PD_DECLARE_int32(pserver_timeout_ms);
PD_DECLARE_int32(heter_world_size);
PD_DECLARE_int32(switch_send_recv_timeout_s);

using MultiVarMsg = MultiVariableMessage;
using VarMsg = VariableMessage;

using serviceHandler =
    std::function<int32_t(const PsRequestMessage& request,
                          PsResponseMessage& response,  // NOLINT
                          brpc::Controller* cntl)>;
using HeterServiceHandler =
    std::function<int32_t(const MultiVarMsg*, MultiVarMsg*, brpc::Controller*)>;

using HeterRpcCallbackFunc = std::function<void(void*)>;

class ServiceHandlerBase {
 public:
  ServiceHandlerBase() : dev_ctx_(nullptr), scope_(nullptr) {}

  virtual ~ServiceHandlerBase() {}

  void SetScope(const framework::Scope* scope) { scope_ = scope; }
  void SetDevCtx(const phi::DeviceContext* dev_ctx) { dev_ctx_ = dev_ctx; }

  virtual int Handle(const MultiVarMsg* request,
                     MultiVarMsg* response,
                     brpc::Controller* cntl) = 0;

 protected:
  const phi::DeviceContext* dev_ctx_;
  const framework::Scope* scope_;
};

using SharedMiniScope =
    std::shared_ptr<std::unordered_map<int, ::paddle::framework::Scope*>>;

using SharedMicroScope = std::shared_ptr<std::unordered_map<
    int,
    std::shared_ptr<std::vector<::paddle::framework::Scope*>>>>;

using SharedTaskQueue = std::shared_ptr<
    std::unordered_map<int,
                       std::shared_ptr<::paddle::framework::BlockingQueue<
                           std::pair<std::string, int>>>>>;

class ValueInSwitch {
 public:
  ValueInSwitch() {}
  ~ValueInSwitch() {}
  char* data() { return _data.data(); }
  size_t size() { return _data.size(); }
  void resize(size_t size) { _data.resize(size); }
  void shrink_to_fit() { _data.shrink_to_fit(); }

 private:
  std::vector<char> _data;
};

class SendAndRecvVariableHandler final : public ServiceHandlerBase {
 public:
  SendAndRecvVariableHandler() {
    this->num_microbatch_ = 0;
    this->num_minibatch_ = 0;
    _local_shards.reset(new shard_type[FLAGS_heter_world_size]);
  }

  virtual ~SendAndRecvVariableHandler() {}

  void SetMiniScopes(SharedMiniScope mini_scopes) {
    mini_scopes_ = mini_scopes;
    num_minibatch_ = mini_scopes_->size();
  }

  void SetMicroScopes(SharedMicroScope micro_scopes) {
    micro_scopes_ = micro_scopes;
    for (auto& scope_pair : (*micro_scopes_)) {
      // auto mini_idx = scope_pair.first;
      auto& micro_scopes = scope_pair.second;
      num_microbatch_ = micro_scopes->size();
      break;
    }
  }

  int GetThreadNum() {
    std::unique_lock<std::mutex> lk(scope_mutex_);
    return (*task_queue_).size();
  }

  int SaveInSwitchWithScope(const MultiVarMsg* request,
                            PsResponseMessage* response,
                            brpc::Controller* cntl);

  void WaitForVarsConsumed(int32_t group_id, const std::string& var_name) {
    // timeline_.Start();
    while (true) {
      {
        std::lock_guard<std::mutex> lock(scope_mutex_);
        if (vars_ready_flag[group_id][var_name] == 0) {
          break;
        }
      }
      /*
      timeline_.Pause();
      if (timeline_.ElapsedSec() > FLAGS_switch_send_recv_timeout_s) {
        VLOG(0) << "vars not consumed exceed 10 minutes";
        break;
      }
      */
    }
    return;
  }

  void WaitForVarsProduced(int32_t group_id, const std::string& var_name) {
    // timeline_.Start();
    while (true) {
      {
        std::lock_guard<std::mutex> lock(scope_mutex_);
        if (vars_ready_flag[group_id][var_name] == 1) {
          break;
        }
      }
      /*
      timeline_.Pause();
      if (timeline_.ElapsedSec() > FLAGS_switch_send_recv_timeout_s) {
        VLOG(0) << "vars not produced exceed 10 minutes";
        break;
      }
      */
    }
    return;
  }

  int SaveInSwitchWithShard(const MultiVarMsg* request,
                            PsResponseMessage* response,
                            brpc::Controller* cntl);

  int QueryInSwitchWithShard(const MultiVarMsg* request,
                             MultiVarMsg* response,
                             brpc::Controller* cntl);

  int QueryInSwitchWithScope(const MultiVarMsg* request,
                             MultiVarMsg* response,
                             brpc::Controller* cntl);

  void SetTaskQueue(SharedTaskQueue task_queue) { task_queue_ = task_queue; }

  int Handle(const MultiVarMsg* request,
             MultiVarMsg* response,
             brpc::Controller* cntl) override {
    LOG(INFO) << "entered Handle";
    phi::RecordEvent record_event("SendAndRecvVariableHandler->Handle",
                                  phi::TracerEventType::Communication,
                                  1);
    FLAGS_eager_delete_tensor_gb = -1;

    // get microID from request
    // deserialize variable to micro scope
    // Push to heter worker's task_queue
    std::unique_ptr<::paddle::framework::Scope> local_scope_ptr(
        new ::paddle::framework::Scope());
    auto& local_scope = *(local_scope_ptr.get());
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    phi::CPUPlace cpu_place;
    auto& cpu_dev_ctx = *pool.Get(cpu_place);

    auto message_name = request->message_name();
    auto& request_io_buffer = cntl->request_attachment();

    distributed::DeserializeFromMultiVarMsgAndIOBuf(
        *request, &request_io_buffer, cpu_dev_ctx, &local_scope);

    auto* var = local_scope.FindVar("microbatch_id");
    PADDLE_ENFORCE_NE(var,
                      nullptr,
                      common::errors::InvalidArgument(
                          "Not find variable microbatch_id in scope."));
    auto* tensor = var->GetMutable<phi::DenseTensor>();
    auto data = reinterpret_cast<const float*>(tensor->data());
    auto micro_id = static_cast<int>(data[0]);
    VLOG(4) << "micro_id in heter server: " << micro_id;
    int minibatch_index = micro_id / 10;
    int microbatch_index = micro_id % 10;

    // check minibatch_index is in mini_scopes_
    std::unique_lock<std::mutex> lk(scope_mutex_);
    if ((*mini_scopes_).find(minibatch_index) != (*mini_scopes_).end()) {
      lk.unlock();

      PADDLE_ENFORCE_EQ(
          (*micro_scopes_).find(minibatch_index) != (*micro_scopes_).end(),
          1,
          common::errors::InvalidArgument(
              "minibatch index should in current trainer"));

    } else {
      // create mini scope & micro scopes
      auto* minibatch_scope = &(scope_->NewScope());
      (*mini_scopes_)[minibatch_index] = minibatch_scope;
      (*micro_scopes_)[minibatch_index].reset(
          new std::vector<::paddle::framework::Scope*>{});
      for (int i = 0; i < num_microbatch_; i++) {
        auto* micro_scope = &(minibatch_scope->NewScope());
        (*((*micro_scopes_)[minibatch_index])).push_back(micro_scope);
      }
      (*task_queue_)[minibatch_index].reset(
          new ::paddle::framework::BlockingQueue<
              std::pair<std::string, int>>());
      lk.unlock();
    }

    auto* micro_scope =
        (*((*micro_scopes_)[minibatch_index]))[microbatch_index];

    distributed::DeserializeFromMultiVarMsgAndIOBuf(
        *request, &request_io_buffer, *dev_ctx_, micro_scope);
    // blocking queue handles multi thread
    VLOG(4) << "Handle in HeterServer: " << message_name << ", "
            << microbatch_index;
    VLOG(4) << "task_queue_ size: " << task_queue_->size();
    (*task_queue_)[minibatch_index]->Push(
        std::make_pair(message_name, microbatch_index));

    auto response_var_nums = request->recv_var_names_size();
    std::vector<std::string> response_var_names(response_var_nums),
        empty_var_names{};
    for (int var_idx = 0; var_idx < response_var_nums; ++var_idx) {
      response_var_names[var_idx] = request->recv_var_names(var_idx);
    }
    auto& response_io_buffer = cntl->response_attachment();
    distributed::SerializeToMultiVarMsgAndIOBuf(message_name,
                                                response_var_names,
                                                empty_var_names,
                                                *dev_ctx_,
                                                &local_scope,
                                                response,
                                                &response_io_buffer);
    VLOG(4) << "Handle over";
    return 0;
  }

 public:
  using shard_type = SparseTableShard<std::string, ValueInSwitch>;
  std::shared_ptr<::paddle::framework::Scope> local_scope_ptr;  // for switch
  std::unordered_map<uint32_t, std::unordered_map<std::string, uint32_t>>
      vars_ready_flag;
  std::unique_ptr<shard_type[]> _local_shards;
  platform::Timer timeline_;

 private:
  // share with HeterPipelineTrainer
  SharedMiniScope mini_scopes_{nullptr};
  SharedMicroScope micro_scopes_{nullptr};

  int num_microbatch_;
  int num_minibatch_;
  std::mutex scope_mutex_;

  bool is_first_stage_ = false;
  bool is_last_stage_ = false;

  SharedTaskQueue task_queue_;
};

class HeterService : public PsService {
 public:
  HeterService() {
    _service_handler_map[PS_STOP_SERVER] =
        std::bind(&HeterService::stop_heter_worker,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3);
    _service_handler_map[PS_START_PROFILER] =
        std::bind(&HeterService::start_profiler,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3);
    _service_handler_map[PS_STOP_PROFILER] =
        std::bind(&HeterService::stop_profiler,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3);

    service_handler_.local_scope_ptr =
        std::make_shared<::paddle::framework::Scope>();
  }

  virtual ~HeterService() {}

  virtual void service(::google::protobuf::RpcController* controller,
                       const PsRequestMessage* request,
                       PsResponseMessage* response,
                       ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);

    response->set_err_code(0);
    response->set_err_msg("");
    brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
    auto itr = _service_handler_map.find(request->cmd_id());
    if (itr == _service_handler_map.end()) {
      std::string err_msg(
          "undefined cmd_id, should match PsCmdID in ps.proto, cmd_id:");
      err_msg.append(std::to_string(request->cmd_id()));
      return;
    }
    serviceHandler handler = itr->second;
    int service_ret = handler(*request, *response, cntl);
    VLOG(4) << "handler in service ret: " << service_ret;
    if (service_ret != 0) {
      response->set_err_code(service_ret);
      response->set_err_msg("server internal error");
    }
  }

  virtual void SendAndRecvVariable(
      ::google::protobuf::RpcController* controller,
      const MultiVarMsg* request,
      MultiVarMsg* response,
      ::google::protobuf::Closure* done) {
    // This object helps you to call done->Run() in RAII style. If you need
    // to process the request asynchronously, pass done_guard.release().
    brpc::ClosureGuard done_guard(done);
    std::string message_name = request->message_name();
    VLOG(0) << "SendAndRecvVariable message_name: " << message_name;
    auto itr = handler_map_.find(message_name);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
    LOG(INFO) << "SendAndRecvVariable(client addr) =" << cntl->remote_side();
    PADDLE_ENFORCE_NE(
        itr,
        handler_map_.end(),
        common::errors::InvalidArgument(
            "HeterService::SendAndRecvVariable Get illegal message_name: %s "
            "which is not in HeterService::handler_map_",
            message_name));
    itr->second(request, response, cntl);
    // We don't want to call done->Run() here, release the guard.
    // done_guard.release();
  }

  virtual void RecvFromSwitch(::google::protobuf::RpcController* controller,
                              const MultiVarMsg* request,
                              MultiVarMsg* response,
                              ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
    // int ret = service_handler_.QueryInSwitchWithScope(request, response,
    // cntl);
    int ret = service_handler_.QueryInSwitchWithShard(request, response, cntl);
    // std::string message_name = request->message_name();
    // auto itr = handler_map_.find(message_name);
    // int ret = itr->second(request, response, cntl);
    if (ret != 0) {
      LOG(ERROR) << "QueryInSwitchWithScope failed!";
    }
    // response->set_message_name(message_name);
  }

  virtual void SendToSwitch(::google::protobuf::RpcController* controller,
                            const MultiVarMsg* request,
                            PsResponseMessage* response,
                            ::google::protobuf::Closure* done) {
    VLOG(4) << "entering SendToSwitch";
    brpc::ClosureGuard done_guard(done);
    std::shared_ptr<HeterClient> switch_client_ptr_ =
        HeterClient::GetSwitchInstance(peer_endpoints_, PEER_ROLE_IS_SWITCH);
    if (switch_client_ptr_->peer_switch_channels_.empty()) {
      LOG(ERROR) << "switch_client_ptr_->peer_switch_channels_ null";
    }
    brpc::Channel* channel = switch_client_ptr_->peer_switch_channels_[0].get();
    brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
    // proxy: 定义新的 OnHeterRpcDone 对象（或者在类 OnHeterRpcDone 中 reset）
    OnHeterRpcDone* closure2 = new OnHeterRpcDone([](void* done) {
      auto* closure = reinterpret_cast<OnHeterRpcDone*>(done);
      int ret = closure->CheckResponse();
      closure->set_promise_value(ret);
      if (closure->cntl.Failed()) {
        PADDLE_ENFORCE_NE(
            closure->cntl.Failed(),
            true,
            common::errors::Unimplemented(
                "HeterClient::SendS2S meets brpc error, error message is %s",
                closure->cntl.ErrorText()));
      }
    });
    auto& std_cntl = closure2->cntl;
    std_cntl.set_timeout_ms(FLAGS_pserver_timeout_ms);
    std_cntl.request_attachment().append(cntl->request_attachment().movable());

    auto promise = std::make_shared<std::promise<int32_t>>();
    closure2->add_promise(promise);
    std::future<int> fut = promise->get_future();
    // brpc::Controller std_cntl;
    // std_cntl.request_attachment().append(cntl->request_attachment().movable());
    PsService_Stub stub(channel);
    stub.SendS2S(&std_cntl, request, response, closure2);
    cntl->response_attachment().append(
        std_cntl.response_attachment().movable());
    fut.wait();
    VLOG(4) << "SendToSwitch done";
    delete closure2;
  }

  void SendS2S(::google::protobuf::RpcController* controller,
               const MultiVarMsg* request,
               PsResponseMessage* response,
               ::google::protobuf::Closure* done) {
    VLOG(4) << "entering SendS2S";
    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
    // int ret = service_handler_.SaveInSwitchWithScope(request, response,
    // cntl);
    int ret = service_handler_.SaveInSwitchWithShard(request, response, cntl);
    // std::string message_name = request->message_name();
    // auto itr = handler_map_.find(message_name);
    // if (itr == handler_map_.end()) {
    //    LOG(ERROR) << "can not find func handler";
    //}
    // int ret = itr->second(request, response, cntl);
    if (ret != 0) {
      LOG(ERROR) << "SaveInSwitchWithScope failed";
    }
    std::string err_msg = "ok";
    response->set_err_msg(err_msg.c_str());
    response->set_err_code(ret);
    VLOG(4) << "heter server SendS2S done";
  }

  void SendToWorker(::google::protobuf::RpcController* controller,
                    const MultiVarMsg* request,
                    PsResponseMessage* response,
                    ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
    VLOG(4) << "SendToWorker(client addr) =" << cntl->remote_side();
    std::shared_ptr<distributed::HeterClient> switch_client_ptr_ =
        HeterClient::GetSwitchInstance(peer_endpoints_, PEER_ROLE_IS_WORKER);
    VLOG(4) << "in switch client, peer worker 0: "
            << switch_client_ptr_->peer_worker_list_[0];
    brpc::Channel* channel = switch_client_ptr_->peer_worker_channels_[0].get();

    auto* closure = reinterpret_cast<OnHeterRpcDone*>(done);
    PsService_Stub stub(channel);
    stub.SendAndRecvVariable(controller, request, &closure->response, done);
    // fill response content
    std::string err_msg("pass to worker");
    response->set_err_msg(err_msg.c_str());
    response->set_err_code(0);
  }

  void RegisterServiceHandler(std::string message_name,
                              HeterServiceHandler func) {
    handler_map_[message_name] = func;
  }

  void SetEndpoint(const std::string& end_point) { endpoint_ = end_point; }

  void SetInterEndpoint(const std::string& end_point) {
    endpoint_inter_ = end_point;
  }

  void SetPeerEndPoints(const std::vector<std::string>& peer_endpoints) {
    peer_endpoints_ = peer_endpoints;
  }

  void SetFanIn(const int& fan_in) { fan_in_ = fan_in; }

  void ForceExit() {
    VLOG(3) << "heter service force exit";
    is_exit_ = true;
    return;
  }

  bool IsExit() { return is_exit_; }

 private:
  int32_t stop_profiler(const PsRequestMessage& request UNUSED,
                        PsResponseMessage& response UNUSED,  // NOLINT
                        brpc::Controller* cntl UNUSED) {
    platform::DisableProfiler(
        platform::EventSortingKey::kDefault,
        string::Sprintf("heter_worker_%s_profile", endpoint_));
    return 0;
  }

  int32_t start_profiler(const PsRequestMessage& request UNUSED,
                         PsResponseMessage& response UNUSED,  // NOLINT
                         brpc::Controller* cntl UNUSED) {
    platform::EnableProfiler(platform::ProfilerState::kAll);
    return 0;
  }

  int32_t stop_heter_worker(const PsRequestMessage& request,
                            PsResponseMessage& response UNUSED,  // NOLINT
                            brpc::Controller* cntl UNUSED) {
    auto client_id = request.client_id();
    stop_cpu_worker_set_.insert(client_id);
    if (stop_cpu_worker_set_.size() == fan_in_) {
      is_exit_ = true;
    }
    return 0;
  }

 private:
  SendAndRecvVariableHandler service_handler_;
  std::string endpoint_;
  std::string endpoint_inter_;
  // for switch
  std::vector<std::string> peer_endpoints_;

  std::unordered_map<int32_t, serviceHandler> _service_handler_map;
  std::unordered_map<std::string, HeterServiceHandler> handler_map_;
  std::unordered_set<int> stop_cpu_worker_set_;
  uint32_t fan_in_;
  bool is_exit_ = false;
};

class HeterServer {
 public:
  HeterServer() : ready_(0) {}
  virtual ~HeterServer() {}
  void Stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_ == true) return;
    if (!IsExit()) {
      service_.ForceExit();
    }
    stopped_ = true;
    cv_.notify_all();
    server_.Stop(1000);
    server_.Join();
  }

  bool IsStop() {
    std::unique_lock<std::mutex> lock(mutex_);
    return stopped_;
  }

  bool IsExit() { return service_.IsExit(); }

  void RegisterServiceHandler(std::string message_name,
                              HeterServiceHandler func);

  void StartHeterService(bool need_encrypt = false);

  void StartHeterInterService(bool need_encrypt = false);

  void SetEndPoint(const std::string& endpoint) {
    this->endpoint_ = endpoint;
    service_.SetEndpoint(endpoint);
  }

  void SetLocalScope() {
    request_handler_->local_scope_ptr =
        std::make_shared<::paddle::framework::Scope>();
  }

  void SetInterEndpoint(const std::string& endpoint) {
    this->endpoint_inter_ = endpoint;
    service_.SetInterEndpoint(endpoint);
  }

  void SetPeerEndPoints(const std::vector<std::string>& peer_endpoints) {
    this->peer_endpoints_ = peer_endpoints;
    service_.SetPeerEndPoints(peer_endpoints);
  }

  void SetFanIn(const int& fan_in);

  void SetServiceHandler(
      std::shared_ptr<SendAndRecvVariableHandler> request_handler) {
    request_handler_ = request_handler;
  }

  void SetMiniBatchScopes(SharedMiniScope mini_scopes) {
    request_handler_->SetMiniScopes(mini_scopes);
  }

  void SetMicroBatchScopes(SharedMicroScope micro_scopes) {
    request_handler_->SetMicroScopes(micro_scopes);
  }

  int GetThreadNum() { return request_handler_->GetThreadNum(); }

  void SetTaskQueue(SharedTaskQueue task_queue) {
    request_handler_->SetTaskQueue(task_queue);
  }

  // HeterWrapper singleton
  static std::shared_ptr<HeterServer> GetInstance() {
    std::unique_lock<std::mutex> lock(mtx_);
    if (s_instance_ == nullptr) {
      s_instance_.reset(new HeterServer());
    }
    return s_instance_;
  }

  void WaitServerReady();

 private:
  static std::shared_ptr<HeterServer> s_instance_;
  mutable std::mutex mutex_;
  static std::mutex mtx_;
  std::condition_variable cv_;
  std::condition_variable condition_ready_;
  bool stopped_ = true;
  std::string endpoint_;
  std::string endpoint_inter_;
  // for switch
  std::vector<std::string> peer_endpoints_;

 protected:
  brpc::Server server_;
  brpc::Server server_inter_;
  HeterService service_;
  std::shared_ptr<SendAndRecvVariableHandler> request_handler_;

  DISABLE_COPY_AND_ASSIGN(HeterServer);
  std::mutex mutex_ready_;

  int ready_;
};

}  // namespace distributed
}  // namespace paddle
