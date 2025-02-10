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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "paddle/phi/core/distributed/store/socket.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/distributed/store/tcp_utils.h"

namespace phi {
namespace distributed {

enum class ReplyType { WAITING, STOP_WAIT, READY, NOT_READY, SUCCESS, FAILURE };
enum class Command { ADD, GET, CHECK, SET, WAIT, DELETEKEY, STOP };

namespace detail {

class MasterDaemon {
 public:
  static std::unique_ptr<MasterDaemon> start(SocketType listen_socket,
                                             int nranks,
                                             int timeout);
  MasterDaemon() = delete;
  explicit MasterDaemon(SocketType listen_socket,
                        int nranks,
                        int stop_check_timeout);
  ~MasterDaemon();

 private:
  void run();
  void ProcessCommands(std::vector<struct pollfd>* p_fds);
  void _do_add(SocketType socket);
  void _do_wait(SocketType socket);
  void _do_get(SocketType socket);
  void _do_check(SocketType socket);
  void _do_set(SocketType socket);
  void _do_deletekey(SocketType socket);
  void _notify_waiting_sockets(const std::string&);
  SocketType _listen_socket;
  std::vector<SocketType> _sockets;
  std::unordered_map<std::string, std::vector<uint8_t>> _store;
  std::thread _background_thread{};
  int _nranks = -1;
  int _timeout = 0;
  std::unordered_map<std::string, std::vector<SocketType>>
      _waiting_sockets;  // key -> list of waiting sockets

  void InitControlFd();
  void CloseControlFd();
  void StopByControlFd();
#ifdef _WIN32
  HANDLE ghStopEvent_{};
#else
  std::array<int, 2> _control_fd{{-1, -1}};
#endif
};

class TCPServer {
 public:
  TCPServer() = default;
  static std::unique_ptr<TCPServer> create(std::uint16_t port,
                                           int nranks,
                                           int stop_check_timeout);

 private:
  std::unique_ptr<MasterDaemon> _master_daemon;
};

class TCPClient {
 public:
  explicit TCPClient(SocketType socket) : _socket{socket} {}
  static std::unique_ptr<TCPClient> connect(const std::string host,
                                            uint16_t port);
  ~TCPClient() { tcputils::close_socket(_socket); }
  void send_command_for_key(Command type, const std::string& key);

  template <typename T>
  void send_value(const T& value);

  template <typename T>
  void send_vector(const std::vector<T>& value);
  template <typename T>
  std::vector<T> receive_vector();

  template <typename T>
  T receive_value();

 private:
  SocketType _socket;
};

}  // namespace detail

// TODO(gongwb) :Add IP6 support.
class TCPStore : public Store {
 public:
  static constexpr std::uint16_t kDefaultPort = 6170;
  explicit TCPStore(std::string host,
                    uint16_t port = kDefaultPort,
                    bool is_master = false,
                    size_t num_workers = 1,
                    int timeout = 900);

  ~TCPStore();

  int64_t add(const std::string& key, int64_t value) override;
  std::vector<uint8_t> get(const std::string& key) override;
  bool check(const std::string& key) override;
  void wait(const std::string& key) override;
  void set(const std::string& key, const std::vector<uint8_t>& value) override;
  bool deleteKey(const std::string& key) override;

 private:
  void waitWorkers();
  std::unique_ptr<detail::TCPServer> _server;
  std::unique_ptr<detail::TCPClient> _client;

  const std::string _init_key = "init/";
  const std::string _key_prefix = "/";

  bool _is_master;
  int _num_workers;
};

}  // namespace distributed
}  // namespace phi
