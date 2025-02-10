// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/store/tcp_store.h"

#include <chrono>
#include <iostream>
#include <thread>

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/phi/core/distributed/store/tcp_utils.h"

namespace phi::distributed::detail {

constexpr int INFTIME = 10000;  // 10 seconds

std::unique_ptr<MasterDaemon> MasterDaemon::start(SocketType socket,
                                                  int nranks,
                                                  int timeout) {
  VLOG(8) << ("begin to run start");
  return std::make_unique<MasterDaemon>(socket, nranks, timeout);
}

MasterDaemon::MasterDaemon(SocketType socket, int nranks, int timeout)
    : _listen_socket(socket), _nranks(nranks), _timeout(timeout) {
  InitControlFd();
  _background_thread = std::thread{&MasterDaemon::run, this};
}

MasterDaemon::~MasterDaemon() {  // NOLINT
  VLOG(8) << ("begin to destruct MasterDaemon");
  StopByControlFd();
  _background_thread.join();
  tcputils::close_socket(_listen_socket);
  for (SocketType socket : _sockets) {
    tcputils::close_socket(socket);
  }
  CloseControlFd();
}

void MasterDaemon::_do_add(SocketType socket) {
  int64_t new_value{};
  std::string key = tcputils::receive_string(socket);
  new_value = tcputils::receive_value<int64_t>(socket);
  std::vector<uint8_t> old_value;
  auto it = _store.find(key);
  if (it != _store.end()) {
    old_value = it->second;
    char* buffer = reinterpret_cast<char*>(it->second.data());
    size_t len = old_value.size();
    new_value += std::stoll(std::string(buffer, len));
  }

  std::string new_value_str = std::to_string(new_value);
  _store[key] =
      std::vector<uint8_t>(new_value_str.begin(), new_value_str.end());
  VLOG(8) << "TCPStore: new value (" << new_value << ") for key (" << key
          << ") " << GetSockName(socket);
  tcputils::send_value<int64_t>(socket, new_value);
  _notify_waiting_sockets(key);
}

void MasterDaemon::_do_set(SocketType socket) {
  std::string key = tcputils::receive_string(socket);
  VLOG(8) << "MasterDaemon::_do_set key(" << key << ") " << GetSockName(socket);

  auto value = tcputils::receive_vector<uint8_t>(socket);
  _store[key] = value;
  _notify_waiting_sockets(key);
}

void MasterDaemon::_notify_waiting_sockets(const std::string& key) {
  if (_waiting_sockets.find(key) != _waiting_sockets.end()) {
    for (auto waiting_socket : _waiting_sockets.at(key)) {
      auto reply = ReplyType::STOP_WAIT;
      VLOG(7) << "TCPStore: notify the socket: " << GetSockName(waiting_socket)
              << " that key: " << key << " is ready.";
      tcputils::send_value<ReplyType>(waiting_socket, reply);
    }
    _waiting_sockets.erase(key);
  }
}

void MasterDaemon::_do_get(SocketType socket) {
  std::string key = tcputils::receive_string(socket);
  VLOG(8) << "MasterDaemon::_do_get key(" << key << ") " << GetSockName(socket);

  auto iter = _store.find(key);
  PADDLE_ENFORCE_NE(
      iter,
      _store.end(),
      common::errors::InvalidArgument("Key %s not found in TCPStore.", key));
  std::vector<uint8_t> value = iter->second;
  tcputils::send_vector<uint8_t>(socket, value);
}

void MasterDaemon::_do_check(SocketType socket) {
  std::string key = tcputils::receive_string(socket);
  VLOG(4) << "MasterDaemon::_do_check key(" << key << ") "
          << GetSockName(socket);

  auto iter = _store.find(key);
  if (iter != _store.end()) {
    tcputils::send_value<ReplyType>(socket, ReplyType::READY);
  } else {
    tcputils::send_value<ReplyType>(socket, ReplyType::NOT_READY);
  }
}

void MasterDaemon::_do_deletekey(SocketType socket) {
  std::string key = tcputils::receive_string(socket);
  VLOG(8) << "MasterDaemon::_do_deletekey key(" << key << ") "
          << GetSockName(socket);

  auto iter = _store.find(key);
  if (iter != _store.end()) {
    _store.erase(iter);
    tcputils::send_value<ReplyType>(socket, ReplyType::SUCCESS);
  } else {
    tcputils::send_value<ReplyType>(socket, ReplyType::FAILURE);
  }
}

#ifndef _WIN32
void MasterDaemon::InitControlFd() {
  PADDLE_ENFORCE_NE(
      pipe(_control_fd.data()),
      -1,
      common::errors::Fatal("failed to cread control pipe errno:%d", errno));
}
void MasterDaemon::CloseControlFd() {
  for (int fd : _control_fd) {
    if (fd != -1) {
      ::close(fd);
    }
  }
}
void MasterDaemon::StopByControlFd() {
  VLOG(8) << ("begin to run StopByControlFd");
  if (_control_fd[1] != -1) {
    PADDLE_ENFORCE_NE(
        ::write(_control_fd[1], "\0", 1),
        -1,
        common::errors::Fatal("failed to write control pipe errno:%d", errno));
    // close the write end of the pipe
    ::close(_control_fd[1]);
    _control_fd[1] = -1;
  }
}
#else
void MasterDaemon::InitControlFd() {
  ghStopEvent_ = CreateEvent(NULL, TRUE, FALSE, NULL);
  PADDLE_ENFORCE_NE(ghStopEvent_,
                    nullptr,
                    common::errors::Fatal("failed to cread control pipe"));
}
void MasterDaemon::CloseControlFd() { CloseHandle(ghStopEvent_); }
void MasterDaemon::StopByControlFd() { SetEvent(ghStopEvent_); }
#endif

void MasterDaemon::_do_wait(SocketType socket) {
  std::string key = tcputils::receive_string(socket);
  VLOG(8) << "MasterDaemon::_do_wait key(" << key << ") "
          << GetSockName(socket);

  auto iter = _store.find(key);
  if (iter == _store.end()) {
    // The key can not be found in store currently. Record and check later.
    _waiting_sockets[key].emplace_back(socket);
  } else {
    auto reply = ReplyType::STOP_WAIT;
    VLOG(7) << "TCPStore: wait reply (" << static_cast<int>(reply)
            << ") for key (" << key << ").";
    tcputils::send_value<ReplyType>(socket, reply);
  }
}

void MasterDaemon::ProcessCommands(std::vector<struct pollfd>* p_fds) {
  std::vector<struct pollfd>& fds = *p_fds;
  // FIXME(gongwb): Don't loop all fds of set just the fds who have event.
#ifdef _WIN32
  // 0: listen socket, so loop from 1.
  for (size_t i = 1; i < fds.size(); i++) {
#else
  // 0: listen socket, 1:controller pipe, so loop from 2.
  for (uint i = 2; i < fds.size(); i++) {
#endif
    try {
      if (fds[i].revents == 0) {
        continue;
      }

      VLOG(8) << "Plan to receive command from " << GetSockName(fds[i].fd);
      Command command = tcputils::receive_value<Command>(fds[i].fd);
      VLOG(7) << "TCPStore: recv command: " << static_cast<int>(command) << ".";

      switch (command) {
        case Command::ADD:
          _do_add(fds[i].fd);
          break;
        case Command::GET:
          _do_get(fds[i].fd);
          break;
        case Command::CHECK:
          _do_check(fds[i].fd);
          break;
        case Command::SET:
          _do_set(fds[i].fd);
          break;
        case Command::WAIT:
          _do_wait(fds[i].fd);
          break;
        case Command::DELETEKEY:
          _do_deletekey(fds[i].fd);
          break;
        default:
          VLOG(8) << "Unknown command: " << static_cast<int>(command)
                  << " from addr info:" << GetSockName(fds[i].fd);
      }
    } catch (const std::exception& ex) {
      auto map_iter = _waiting_sockets.begin();
      while (map_iter != _waiting_sockets.end()) {
        auto vec_iter = map_iter->second.begin();
        while (vec_iter != map_iter->second.end()) {
          if (*vec_iter == fds[i].fd) {
            vec_iter = map_iter->second.erase(vec_iter);
          } else {
            ++vec_iter;
          }
        }
        if (map_iter->second.empty()) {
          map_iter = _waiting_sockets.erase(map_iter);
        } else {
          ++map_iter;
        }
      }

      tcputils::close_socket(fds[i].fd);
      fds.erase(fds.begin() + i);
#ifdef _WIN32
      _sockets.erase(_sockets.begin() + i - 1);
#else
      _sockets.erase(_sockets.begin() + i - 2);
#endif
      std::string s(ex.what());
      if (s.find("TCP connection reset by peer") != std::string::npos) {
        VLOG(5) << "TCP connection reset by peer";
      } else {
        VLOG(5) << "Meet some exceptions during run:" << ex.what();
      }
    }
  }
}

void MasterDaemon::run() {
  std::vector<struct pollfd> fds;
#ifdef _WIN32
  fds.push_back({_listen_socket, POLLIN});
#else
  fds.push_back({.fd = _listen_socket, .events = POLLIN, .revents = 0});
  fds.push_back(
      {.fd = _control_fd[0], .events = POLLIN | POLLHUP, .revents = 0});
#endif

  bool finished = false;
  while (!finished) {
    for (auto& item : fds) {
      item.revents = 0;
    }

    VLOG(9) << "begin to poll fds_size:"
            << paddle::string::Sprintf("%d", fds.size());
#ifdef _WIN32
    int res = ::WSAPoll(fds.data(), fds.size(), INFTIME);
    if (res == 0) {
      auto rv = WaitForSingleObject(ghStopEvent_, 0);
      if (rv != WAIT_TIMEOUT) {
        finished = true;
        break;
      }
      continue;
    }
#else
    ::poll(fds.data(), fds.size(), INFTIME);

    VLOG(9) << "begin to fds[1].revents:"
            << paddle::string::Sprintf("%d", fds[1].revents);
    // The control pipe receive shutdown event, and begin to close it.
    if (fds[1].revents != 0) {
      if (fds[1].revents & ~(POLLIN | POLLHUP)) {
        PADDLE_THROW(
            common::errors::Fatal("Undefined event type:%d", fds[1].revents));
      }
      VLOG(0)
          << "receive shutdown event and so quit from MasterDaemon run loop";
      finished = true;  // NOLINT
      break;
    }
#endif

    // accept connect request.
    if (fds[0].revents != 0) {
      auto socket = tcputils::tcp_accept(_listen_socket);
      _sockets.emplace_back(socket);
#ifdef _WIN32
      fds.push_back({socket, POLLIN});
#else
      fds.push_back({.fd = socket, .events = POLLIN, .revents = 0});
#endif
    }

    ProcessCommands(&fds);
  }
}

std::unique_ptr<TCPServer> TCPServer::create(uint16_t port,
                                             int nranks,
                                             int stop_check_timeout) {
  int socket = tcputils::tcp_listen("", std::to_string(port), AF_INET);
  auto server = std::make_unique<TCPServer>();
  server->_master_daemon =
      MasterDaemon::start(socket, nranks, stop_check_timeout);
  return server;
}

std::unique_ptr<TCPClient> TCPClient::connect(const std::string host,
                                              uint16_t port) {
  int socket = tcputils::tcp_connect(host, std::to_string(port), AF_INET);
  return std::make_unique<TCPClient>(socket);
}

void TCPClient::send_command_for_key(Command type, const std::string& key) {
  tcputils::send_value<Command>(_socket, type);
  if (key.empty()) {
    return;
  }
  tcputils::send_string(_socket, key);
}

template <typename T>
void TCPClient::send_value(const T& value) {
  tcputils::send_bytes<T>(_socket, &value, 1);
}

template <typename T>
T TCPClient::receive_value() {
  T res;
  tcputils::receive_bytes<T>(_socket, &res, 1);
  return res;
}

template <typename T>
void TCPClient::send_vector(const std::vector<T>& value) {
  tcputils::send_vector<T>(_socket, value);
}

template <typename T>
std::vector<T> TCPClient::receive_vector() {
  return tcputils::receive_vector<T>(_socket);
}

}  // namespace phi::distributed::detail
namespace phi::distributed {

TCPStore::TCPStore(std::string host,
                   uint16_t port,
                   bool is_master,
                   size_t num_workers,
                   int timeout)
    : Store(timeout),
      _is_master(is_master),
      _num_workers(static_cast<int>(num_workers)) {
  _timeout = timeout;
  PADDLE_ENFORCE_GT(
      timeout,
      0,
      common::errors::InvalidArgument("timeout must >= %d", timeout));

  VLOG(7) << "input timeout" << timeout << ", member timeout:" << _timeout;
  if (_is_master) {
    _server = detail::TCPServer::create(port, this->_num_workers, timeout);
  }

  _client = detail::TCPClient::connect(host, port);
  waitWorkers();
}

void TCPStore::waitWorkers() {
  if (_num_workers == 0) {
    return;
  }
  add(_init_key, 1);

  if (_is_master) {
    VLOG(7) << paddle::string::Sprintf("_timeout:%d", _timeout);
    auto begin = std::chrono::steady_clock::now();
    do {
      auto value = get(_init_key);
      int completed = std::stoi(std::string(value.begin(), value.end()));
      VLOG(7) << completed << " worker ready, total " << _num_workers
              << ", _timeout:" << _timeout;
      if (completed >= _num_workers) {
        break;
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - begin);

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (_timeout != 0 && elapsed.count() > _timeout) {
        PADDLE_THROW(common::errors::Fatal(paddle::string::Sprintf(
            "_timeout:%d elapsed:%d (elapsed > _timeout)=%d",
            _timeout,
            elapsed.count(),
            elapsed.count() > _timeout)));

        PADDLE_ENFORCE_EQ(
            completed,
            _num_workers,
            common::errors::InvalidArgument(
                "TCPStore timeouted and not all workers got ready."));
      }
    } while (true);
  }
  VLOG(7) << "TCPStore initialized.";
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  VLOG(7) << "TCPStore add.";
  _client->send_command_for_key(Command::ADD, _key_prefix + key);
  _client->send_value<std::int64_t>(value);
  return _client->receive_value<std::int64_t>();
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  VLOG(7) << "TCPStore set.";
  _client->send_command_for_key(Command::SET, _key_prefix + key);
  _client->send_vector<uint8_t>(value);
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  wait(key);
  _client->send_command_for_key(Command::GET, _key_prefix + key);
  VLOG(7) << "TCPStore get.";
  return _client->receive_vector<uint8_t>();
}

bool TCPStore::check(const std::string& key) {
  _client->send_command_for_key(Command::CHECK, _key_prefix + key);
  VLOG(3) << "TCPStore check.";
  auto response = _client->receive_value<ReplyType>();
  if (response == ReplyType::READY) {
    return true;
  } else {
    return false;
  }
}

void TCPStore::wait(const std::string& key) {
  ReplyType reply;  // NOLINT
  VLOG(7) << "TCPStore wait.";
  _client->send_command_for_key(Command::WAIT, _key_prefix + key);
  reply = _client->receive_value<ReplyType>();
  PADDLE_ENFORCE_EQ(
      reply == ReplyType::STOP_WAIT,
      true,
      common::errors::InvalidArgument("Stop_waiting response is expected"));
}

bool TCPStore::deleteKey(const std::string& key) {
  _client->send_command_for_key(Command::DELETEKEY, _key_prefix + key);
  VLOG(3) << "TCPStore deleteKey.";
  auto response = _client->receive_value<ReplyType>();
  if (response == ReplyType::SUCCESS) {
    return true;
  } else {
    return false;
  }
}

TCPStore::~TCPStore() { VLOG(7) << "TCPStore destructure"; }

}  // namespace phi::distributed
