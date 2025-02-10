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

#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "butil/endpoint.h"
#include "google/protobuf/service.h"
#include "paddle/common/macros.h"
#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/ps/service/env.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_context.h"

namespace google {
namespace protobuf {
class RpcController;
}  // namespace protobuf
}  // namespace google
namespace paddle {
namespace distributed {
class PSEnvironment;
}  // namespace distributed
}  // namespace paddle

namespace paddle {
namespace framework {
class Executor;
class ProgramDesc;
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace distributed {

class Table;

using ::paddle::distributed::PsRequestMessage;
using ::paddle::distributed::PsResponseMessage;

class PSServer {
 public:
  PSServer() {}
  virtual ~PSServer() {}
  PSServer(PSServer &&) = delete;
  PSServer(const PSServer &) = delete;

  virtual int32_t Configure(
      const PSParameter &config,
      PSEnvironment &env,  // NOLINT
      size_t server_rank,
      const std::vector<framework::ProgramDesc> &server_sub_program = {});

  virtual uint64_t Start(const std::string &ip, uint32_t port) = 0;
  virtual int32_t Stop() = 0;

  inline size_t Rank() const { return _rank; }

  inline PSEnvironment *Environment() { return _environment; }

  inline const ServerParameter *Config() const { return &_config; }
  inline Table *GetTable(size_t table_id) {
    auto itr = _table_map.find(table_id);
    if (itr != _table_map.end()) {
      return itr->second.get();
    }
    return NULL;
  }

  inline std::unordered_map<uint32_t, std::shared_ptr<Table>> *GetTable() {
    return &_table_map;
  }

  // for cache
  virtual int32_t StartS2S() { return 0; }

  virtual ::std::future<int32_t> SendPServer2PServerMsg(
      int msg_type UNUSED,
      int to_pserver_id UNUSED,
      const std::string &msg UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "NotImplementError: PSServer::send_pserver2pserver_msg"));
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  typedef std::function<int32_t(int, int, const std::string &)> MsgHandlerFunc;
  virtual int RegisterPServer2PServerMsgHandler(int msg_type,
                                                MsgHandlerFunc handler) {
    _msg_handler_map[msg_type] = handler;
    return 0;
  }
  virtual int HandlePServer2PServerMsg(int msg_type,
                                       int from_pserver_id,
                                       const std::string &msg) {
    auto itr = _msg_handler_map.find(msg_type);
    if (itr == _msg_handler_map.end()) {
      if (msg_type == 101) {
        return ReceiveFromPServer(msg_type, from_pserver_id, msg);
      } else {
        LOG(WARNING) << "unknown pserver2pserver_msg type:" << msg_type;
        return -1;
      }
    }
    return itr->second(msg_type, from_pserver_id, msg);
  }
  virtual int32_t ReceiveFromPServer(int msg_type UNUSED,
                                     int pserver_id UNUSED,
                                     const std::string &msg UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "NotImplementError::PSServer::ReceiveFromPServer"));
    return -1;
  }

  ::paddle::framework::Channel<std::pair<uint64_t, std::string>> _shuffled_ins;

 protected:
  virtual int32_t Initialize() = 0;

 protected:
  size_t _rank;
  ServerParameter _config;
  PSEnvironment *_environment;
  std::unordered_map<uint32_t, std::shared_ptr<Table>> _table_map;
  std::unordered_map<int32_t, MsgHandlerFunc> _msg_handler_map;

 protected:
  std::shared_ptr<framework::Scope> scope_;
  phi::Place place_ = phi::CPUPlace();
};

REGISTER_PSCORE_REGISTERER(PSServer);

typedef std::function<void(void *)> PServerCallBack;

class PServerClosure : public google::protobuf::Closure {
 public:
  explicit PServerClosure(PServerCallBack callback) : _callback(callback) {}
  virtual ~PServerClosure() {}
  virtual void set_promise_value(int value) {
    for (auto &promise : _promises) {
      promise->set_value(value);
    }
  }
  void add_promise(const std::shared_ptr<std::promise<int32_t>> &promise) {
    _promises.push_back(promise);
  }

 protected:
  PServerCallBack _callback;
  std::vector<std::shared_ptr<std::promise<int32_t>>> _promises;
};

class PsBaseService : public PsService {
 public:
  PsBaseService() : _rank(0), _server(NULL), _config(NULL) {}
  virtual ~PsBaseService() {}
  virtual size_t GetRank() { return _rank; }
  virtual int32_t Configure(PSServer *server) {
    _server = server;
    _rank = _server->Rank();
    _config = _server->Config();
    return 0;
  }
  void service(::google::protobuf::RpcController *controller,
               const PsRequestMessage *request,
               PsResponseMessage *response,
               ::google::protobuf::Closure *done) override = 0;

  virtual void set_response_code(PsResponseMessage &response,  // NOLINT
                                 int err_code,
                                 const char *err_msg) {
    response.set_err_msg(err_msg);
    response.set_err_code(err_code);
    LOG(WARNING) << "Response err_code:" << err_code << " msg:" << err_msg;
  }

  virtual int32_t Initialize() = 0;
  PSServer *GetServer() { return _server; }

 protected:
  size_t _rank;
  PSServer *_server;
  const ServerParameter *_config;
};
REGISTER_PSCORE_REGISTERER(PsBaseService);

class PSServerFactory {
 public:
  static PSServer *Create(const PSParameter &config);
};
}  // namespace distributed
}  // namespace paddle
