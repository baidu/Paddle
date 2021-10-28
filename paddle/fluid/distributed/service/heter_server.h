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
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/service/brpc_utils.h"
#include "paddle/fluid/distributed/service/sendrecv.pb.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN
#include "paddle/fluid/platform/profiler.h"

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
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

DECLARE_double(eager_delete_tensor_gb);
namespace paddle {
namespace distributed {

static void split(const std::string& str, char sep,
                  std::vector<std::string>* pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;
using VarMsg = ::paddle::distributed::VariableMessage;

class HeterService;

typedef int32_t (HeterService::*serviceHandlerFunc)(
    const PsRequestMessage& request, PsResponseMessage& response,
    brpc::Controller* cntl);

typedef std::function<void(void*)> HeterRpcCallbackFunc;
typedef std::function<int(const MultiVarMsg*, MultiVarMsg*, brpc::Controller*)>
    HeterServiceHandler;

class HeterService : public ::paddle::distributed::PsService {
 public:
  HeterService() {
    _service_handler_map[PS_STOP_SERVER] = &HeterService::stop_heter_worker;
    _service_handler_map[PS_START_PROFILER] = &HeterService::start_profiler;
    _service_handler_map[PS_STOP_PROFILER] = &HeterService::stop_profiler;
  }

  virtual ~HeterService() {}

  virtual void service(::google::protobuf::RpcController* controller,
                       const PsRequestMessage* request,
                       PsResponseMessage* response,
                       ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    std::string log_label("ReceiveCmd-");

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
    serviceHandlerFunc handler_func = itr->second;
    int service_ret = (this->*handler_func)(*request, *response, cntl);
    if (service_ret != 0) {
      response->set_err_code(service_ret);
      response->set_err_msg("server internal error");
    }
  }

  void SendAndRecvVariable(::google::protobuf::RpcController* controller,
                           const MultiVarMsg* request, MultiVarMsg* response,
                           ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    std::string message_name = request->message_name();
    auto itr = handler_map_.find(message_name);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
    PADDLE_ENFORCE_NE(
        itr, handler_map_.end(),
        platform::errors::InvalidArgument(
            "HeterService::SendAndRecvVariable Get illegal message_name: %s "
            "which is not in HeterService::handler_map_",
            message_name));
    itr->second(request, response, cntl);
  }

  void RegisterServiceHandler(std::string message_name,
                              HeterServiceHandler func) {
    handler_map_[message_name] = func;
  }

  void SetEndpoint(const std::string& end_point) { endpoint_ = end_point; }
  void SetFanin(const int& fan_in) { fan_in_ = fan_in; }
  bool IsExit() { return is_exit_; }

 private:
  int32_t stop_profiler(const PsRequestMessage& request,
                        PsResponseMessage& response, brpc::Controller* cntl);

  int32_t start_profiler(const PsRequestMessage& request,
                         PsResponseMessage& response, brpc::Controller* cntl);

  int32_t stop_heter_worker(const PsRequestMessage& request,
                            PsResponseMessage& response,
                            brpc::Controller* cntl);

 private:
  std::string endpoint_;
  std::unordered_map<std::string, HeterServiceHandler> handler_map_;
  std::unordered_map<int32_t, serviceHandlerFunc> _service_handler_map;
  std::unordered_set<int> stop_cpu_worker_set_;
  int fan_in_;
  bool is_exit_ = false;
};


using SharedMiniScope = std::shared_ptr<std::unordered_map<int, ::paddle::framework::Scope*>>;
using SharedMicroScope = std::shared_ptr<std::unordered_map<int, std::shared_ptr<std::vector<::paddle::framework::Scope*>> > >;

using SharedTaskQueue = std::shared_ptr<std::unordered_map<int, std::shared_ptr<
                        ::paddle::framework::BlockingQueue<std::pair<std::string, int>>> >>;

class HeterRequestHandler {
 public:
  HeterRequestHandler()
      : dev_ctx_(nullptr),
        scope_(nullptr),
        program_(nullptr) {}

  virtual ~HeterRequestHandler() {}

  void SetScope(const framework::Scope* scope) { scope_ = scope; }
  void SetDevCtx(const platform::DeviceContext* dev_ctx) { dev_ctx_ = dev_ctx; }
  void SetProgram(framework::ProgramDesc* program) { program_ = program; }

  virtual int Handle(const MultiVarMsg* request, MultiVarMsg* response,
                     brpc::Controller* cntl) = 0;

 protected:
  const platform::DeviceContext* dev_ctx_;
  const framework::Scope* scope_;
  framework::ProgramDesc* program_;
};

class RequestSendAndRecvHandler final : public HeterRequestHandler {
 public:
  RequestSendAndRecvHandler() {
    this->num_microbatch_ = 0;
    this->num_minibatch_ = 0;
  }

  virtual ~RequestSendAndRecvHandler() { 
  }

  void SetMiniScopes(SharedMiniScope mini_scopes) {
      mini_scopes_ = mini_scopes;
      num_minibatch_ = mini_scopes_->size();
  }
  void SetMicroScopes(SharedMicroScope micro_scopes) {
      micro_scopes_ = micro_scopes;
      num_microbatch_ = micro_scopes_->size();
  }

  void SetTaskQueue(SharedTaskQueue task_queue) {
      task_queue_ = task_queue;    
  }
  
  int Handle(const MultiVarMsg* request, MultiVarMsg* response,
             brpc::Controller* cntl) override {
    platform::RecordEvent record_event("RequestSendAndRecvHandler->Handle");
    FLAGS_eager_delete_tensor_gb = -1;
    
    // get microID from request
    // deserialize variable to micro scope
    // Push to heter worker's task_queue
    //auto& local_scope = scope_->NewScope();
    std::unique_ptr<paddle::framework::Scope> local_scope_ptr(new paddle::framework::Scope());
    auto& local_scope = *(local_scope_ptr.get());
    auto message_name = request->message_name();
      auto& request_io_buffer = cntl->request_attachment();
      distributed::DeserializeFromMultiVarMsgAndIOBuf(
          *request, &request_io_buffer, *dev_ctx_, &local_scope);

      auto* var = local_scope.FindVar("microbatch_id");
      PADDLE_ENFORCE_NE(var, nullptr,
                        platform::errors::InvalidArgument(
                            "Not find variable microbatch_id in scope."));
      auto* tensor = var->GetMutable<framework::LoDTensor>();
      const auto place = dev_ctx_->GetPlace()

      int micro_id = -1;
      if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
        char* temp_ptr =
            new char[tensor->numel() * framework::SizeOfType(tensor->type())];
        auto stream =
            reinterpret_cast<const platform::CUDADeviceContext&>(*dev_ctx_)
                .stream();
        memory::Copy(platform::CPUPlace(), temp_ptr,
                     BOOST_GET_CONST(platform::CUDAPlace, tensor->place()),
                     tensor->data<void>(),
                     tensor->numel() * framework::SizeOfType(tensor->type()),
                     stream);
        float* temp_ptr_float = reinterpret_cast<float*>(temp_ptr);
        micro_id = static_cast<int>(temp_ptr_float[0]);
        delete[] temp_ptr;
#endif
      } else {
        auto data = reinterpret_cast<const float*>(tensor->data<void>());
        micro_id = static_cast<int>(data[0]);
      }

      int minibatch_index = micro_id / 10;
      int microbatch_index = micro_id % 10;
      
      PADDLE_ENFORCE_EQ((*mini_scopes_).find(minibatch_index) != (*mini_scopes_).end(), 1,
                        platform::errors::InvalidArgument(
                            "minibatch index should in current trainer"));
	  PADDLE_ENFORCE_EQ((*micro_scopes_).find(minibatch_index) != (*micro_scopes_).end(), 1,
						platform::errors::InvalidArgument(
							"minibatch index should in current trainer"));
      
      auto* micro_scope = (*((*micro_scopes_)[minibatch_index]))[microbatch_index];

      distributed::DeserializeFromMultiVarMsgAndIOBuf(
          *request, &request_io_buffer, *dev_ctx_, micro_scope);
      // blocking queue handles multi thread
      (*task_queue_)[minibatch_index]->Push(std::make_pair(message_name, microbatch_index));

    auto response_var_nums = request->recv_var_names_size();
    std::vector<std::string> response_var_names(response_var_nums),
        empty_var_names{};
    for (int var_idx = 0; var_idx < response_var_nums; ++var_idx) {
      response_var_names[var_idx] = request->recv_var_names(var_idx);
    }
    auto& response_io_buffer = cntl->response_attachment();
    distributed::SerializeToMultiVarMsgAndIOBuf(
        message_name, response_var_names, empty_var_names, *dev_ctx_,
        &local_scope, response, &response_io_buffer);
    return 0;
  }

 private:
  // share with HeterPipelineTrainer
  SharedMiniScope mini_scopes_{nullptr};
  SharedMicroScope micro_scopes_{nullptr};
  
  int num_microbatch_;
  int num_minibatch_;

  bool is_first_stage_ = false;
  bool is_last_stage_ = false;

  SharedTaskQueue task_queue_;

};

class HeterServer {
 public:
  virtual ~HeterServer() {}

  void Stop() {
    VLOG(3) << "HeterServer Stop()";
    std::unique_lock<std::mutex> lock(mutex_);
    stoped_ = true;
    cv_.notify_all();
    server_.Stop(1000);
    server_.Join();
  }

  bool IsExit() { return service_.IsExit(); }

  HeterServer() {}

  void RegisterServiceHandler(std::string message_name,
                              HeterServiceHandler func);

  void StartHeterService();

  void SetEndPoint(std::string& endpoint);
  void SetFanin(int& fan_in);

  void SetRequestHandler(std::shared_ptr<RequestSendAndRecvHandler> request_handler) { 
      request_handler_ = request_handler; 
  }

  void SetMiniBatchScopes(SharedMiniScope  mini_scopes) {
      request_handler_->SetMiniScopes(mini_scopes);
  }
  void SetMicroBatchScopes(SharedMicroScope micro_scopes) {
      request_handler_->SetMicroScopes(micro_scopes);
  }

  void SetTaskQueue(SharedTaskQueue task_queue) {
      request_handler_->SetTaskQueue(task_queue);
  }

  // HeterWrapper singleton
  static std::shared_ptr<HeterServer> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new HeterServer());
    }
    return s_instance_;
  }

  void WaitServerReady();

 private:
  static std::shared_ptr<HeterServer> s_instance_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable condition_ready_;
  bool stoped_ = false;
  std::string endpoint_;

 protected:
  brpc::Server server_;
  HeterService service_;
  std::shared_ptr<RequestSendAndRecvHandler> request_handler_;

  DISABLE_COPY_AND_ASSIGN(HeterServer);
  std::mutex mutex_ready_;

  int ready_;
};


}  // end namespace distributed
}  // end namespace paddle
