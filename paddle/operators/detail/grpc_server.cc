/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "grpc_server.h"
#include <future>

using grpc::ServerAsyncResponseWriter;

namespace paddle {
namespace operators {
namespace detail {

enum CallStatus { CREATE, PROCESS, FINISH };

// reference:
// https://stackoverflow.com/questions/41732884/grpc-multiple-services-in-cpp-async-server
class RequestBase {
 public:
  virtual void Proceed() = 0;
};

class RequestSend final : public RequestBase {
 public:
  RequestSend(SendRecvService::AsyncService* service, ServerCompletionQueue* cq)
      : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
    Proceed();
  }

  void Proceed() {
    if (status_ == CREATE) {
      status_ = PROCESS;

      service_->RequestSendVariable(&ctx_, &request_, &responder_, cq_, cq_,
                                    this);
    } else if (status_ == PROCESS) {
      new RequestSend(service_, cq_);

      // The actual processing.
      // std::string prefix("Hello ");
      // reply_.set_message(prefix + request_.name());

      status_ = FINISH;
      responder_.Finish(reply_, Status::OK, this);
    } else {
      GPR_ASSERT(status_ == FINISH);
      delete this;
    }
  }

 private:
  VariableMessage request_;
  VoidMessage reply_;
  ServerContext ctx_;

  SendRecvService::AsyncService* service_;
  ServerCompletionQueue* cq_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
  CallStatus status_;
};

class RequestGet final : public RequestBase {
 public:
  RequestGet(SendRecvService::AsyncService* service, ServerCompletionQueue* cq)
      : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
    Proceed();
  }

  void Proceed() {
    // wait to this server has receive all gradients.
    if (status_ == CREATE) {
      status_ = PROCESS;

      service_->RequestGetVariable(&ctx_, &request_, &responder_, cq_, cq_,
                                   this);
    } else if (status_ == PROCESS) {
      new RequestGet(service_, cq_);

      // The actual processing.
      // std::string prefix("Hello ");
      // reply_.set_message(prefix + request_.name());

      status_ = FINISH;
      responder_.Finish(reply_, Status::OK, this);
    } else {
      GPR_ASSERT(status_ == FINISH);
      delete this;
    }
  }

 private:
  VariableMessage request_;
  VariableMessage reply_;
  ServerContext ctx_;

  SendRecvService::AsyncService* service_;
  ServerCompletionQueue* cq_;
  ServerAsyncResponseWriter<VariableMessage> responder_;
  CallStatus status_;
};

void AsyncGRPCServer::RunSyncUpdate() {
  ServerBuilder builder;
  builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
  builder.RegisterService(&service_);

  cq_send_ = builder.AddCompletionQueue();
  cq_get_ = builder.AddCompletionQueue();
  server_ = builder.BuildAndStart();
  // std::cout << "Server listening on " << server_address << std::endl;

  t_send_.reset(
      new std::thread(std::bind(&AsyncGRPCServer::_HandleReqSend, this)));
  t_get_.reset(
      new std::thread(std::bind(&AsyncGRPCServer::_HandleReqGet, this, true)));

  // wait server
  server_->Wait();
}

void AsyncGRPCServer::ShutDown() {
  exit_ = true;
  cq_send_->Shutdown();
  cq_get_->Shutdown();
  t_send_->join();
  t_get_->join();
  server_->Shutdown();
}

void AsyncGRPCServer::_HandleReqSend() {
  std::unique_ptr<RequestSend> req_send(
      new RequestSend(&service_, cq_send_.get()));
  void* tag = NULL;
  bool ok = false;
  while (true) {
    if (cq_send_->Next(&tag, &ok)) {
      break;
    }
    if (!ok) {
      continue;
    }
    static_cast<RequestBase*>(tag)->Proceed();
  }
}

void AsyncGRPCServer::_HandleReqGet(bool wait) {
  std::unique_ptr<RequestGet> req_get(new RequestGet(&service_, cq_get_.get()));
  void* tag = NULL;
  bool ok = false;
  while (true) {
    if (!cq_get_->Next(&tag, &ok)) {
      break;
    }
    if (!ok) {
      continue;
    }

    if (wait && !done_) {
      _Wait();
    }
    static_cast<RequestBase*>(tag)->Proceed();
  }
}

void AsyncGRPCServer::_Wait() {
  std::unique_lock<std::mutex> lock(this->mutex_);
  condition_.wait(lock, [=] { return this->done_ == true; });
}

Status AsyncGRPCServer::Wait(ServerContext* context, const VoidMessage* in_var,
                             VoidMessage* out_var) {
  _Wait();
  return Status::OK;
}

void AsyncGRPCServer::Reset() {
  std::lock_guard<std::mutex> lock(this->mutex_);
  done_ = false;
}

void AsyncGRPCServer::Done() {
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    done_ = true;
  }
  condition_.notify_all();
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
