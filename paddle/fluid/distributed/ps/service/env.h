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

#include <arpa/inet.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <stdio.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/common/macros.h"

namespace paddle {
namespace distributed {

struct PSHost {
  std::string ip;
  uint32_t port;
  uint32_t rank;

  PSHost() = default;
  PSHost(const std::string ip, uint32_t port, uint32_t rank)
      : ip(ip), port(port), rank(rank) {}

  // |---ip---|---port---|--rank--|
  // |-32bit--|--20bit---|--12bit-|

  uint64_t SerializeToUint64() {
    uint64_t host_label = 0;
    host_label = inet_addr(ip.c_str());
    host_label = host_label << 32;
    host_label += (port << 12);
    host_label += rank;
    return host_label;
  }

  void ParseFromUint64(uint64_t host_label) {
    static uint64_t rank_label_mask = (1L << 12) - 1;
    static uint64_t port_label_mask = (1L << 20) - 1;
    rank = host_label & rank_label_mask;
    port = (host_label >> 12) & port_label_mask;
    uint32_t ip_addr = (host_label >> 32);
    ip = inet_ntoa(*(in_addr *)&ip_addr);  // NOLINT
  }

  std::string ToString() {
    std::stringstream s;
    s << "host: " << ip;
    s << " port: " << port;
    s << " rank: " << rank;
    s << " uint64: " << SerializeToUint64();
    return s.str();
  }

  // for open source parameter server
  std::string SerializeToString() {
    std::stringstream s;
    s << ip << ":";
    s << port << ":";
    s << rank;
    return s.str();
  }

  void ParseFromString(std::string endpoint) {
    std::vector<std::string> endpoint_info;
    StringSplit(endpoint, ':', &endpoint_info);
    ip = endpoint_info[0];
    port = std::stoi(endpoint_info[1]);
    rank = std::stoi(endpoint_info[2]);
  }

  void StringSplit(const std::string &str,
                   char sep,
                   std::vector<std::string> *pieces,
                   bool ignore_null = true) {
    pieces->clear();
    if (str.empty()) {
      if (!ignore_null) {
        pieces->push_back(str);
      }
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
};

class PSEnvironment {
 public:
  explicit PSEnvironment() {}  // NOLINT
  virtual ~PSEnvironment() {}

  virtual int32_t SetPsServers(uint64_t *host_sign_list UNUSED,
                               int node_num UNUSED) {
    return 0;
  }
  virtual int32_t SetPsServers(
      const std::vector<std::string> *host_endpoint_list UNUSED,
      int node_num UNUSED) {
    return 0;
  }

  virtual int32_t SetPsClients(uint64_t *host_sign_list UNUSED,
                               int node_num UNUSED) {
    return 0;
  }

  virtual int32_t SetPsClients(std::string *host_endpoint_list UNUSED,
                               int node_num UNUSED) {
    return 0;
  }

  virtual uint64_t GetLocalHostSign() { return 0; }
  virtual std::vector<PSHost> GetPsServers() const { return _ps_server_list; }
  virtual int32_t RegisterPsServer(const std::string &ip,
                                   uint32_t port,
                                   int32_t rank) {
    return RegisterPsHost(ip, port, rank, _ps_server_list, _ps_server_sign_set);
  }

  virtual std::vector<PSHost> GetPsClients() const { return _ps_client_list; }
  virtual int32_t RegisterPsClient(const std::string &ip,
                                   uint32_t port,
                                   int32_t rank) {
    return RegisterPsHost(ip, port, rank, _ps_client_list, _ps_client_sign_set);
  }

  virtual std::vector<PSHost> GetCoordinators() const {
    return _coordinator_list;
  }
  virtual int32_t RegisterCoordinatorClient(const std::string &ip,
                                            uint32_t port,
                                            int32_t rank) {
    return RegisterPsHost(
        ip, port, rank, _coordinator_list, _coordinator_sign_set);
  }

  virtual std::vector<uint64_t> GetClientInfo() {
    std::vector<uint64_t> client_info;
    for (auto &i : _ps_client_list) {
      client_info.push_back(i.SerializeToUint64());
    }
    return client_info;
  }

  virtual std::vector<std::string> GetClientInfo(bool use_string_endpoint) {
    if (use_string_endpoint) {
      std::vector<std::string> client_info;
      for (auto &i : _ps_client_list) {
        client_info.push_back(i.SerializeToString());
      }
      return client_info;
    }
    return {};
  }

  virtual void SetTrainers(int trainers) { trainers_ = trainers; }

  virtual int GetTrainers() { return trainers_; }

 protected:
  //注册一个host //  NOLINT
  virtual int32_t RegisterPsHost(
      const std::string &ip,
      uint32_t port,
      int32_t rank,
      std::vector<PSHost> &host_list,            // NOLINT
      std::unordered_set<uint64_t> &sign_set) {  // NOLINT
    PSHost host;
    host.ip = ip;
    host.port = port;
    host.rank = rank;

    if (sign_set.count(rank) == 0) {
      host_list.push_back(host);
      sign_set.insert(rank);
    }

    return 0;
  }

  int trainers_ = 0;

  std::vector<PSHost> _ps_client_list;
  std::unordered_set<uint64_t> _ps_client_sign_set;  // for unique filter

  std::vector<PSHost> _ps_server_list;
  std::unordered_set<uint64_t> _ps_server_sign_set;  // for unique filter

  std::vector<PSHost> _coordinator_list;
  std::unordered_set<uint64_t> _coordinator_sign_set;
};

class PaddlePSEnvironment : public PSEnvironment {
 public:
  explicit PaddlePSEnvironment() {}  // NOLINT
  virtual ~PaddlePSEnvironment() {}

  virtual int32_t SetPsServers(uint64_t *host_sign_list, int node_num) {
    _ps_server_list.clear();
    _ps_server_sign_set.clear();
    for (int i = 0; i < node_num; ++i) {
      if (host_sign_list[i] > 0) {
        PSHost host;
        host.ParseFromUint64(host_sign_list[i]);
        _ps_server_list.push_back(host);
        _ps_server_sign_set.insert(host.SerializeToUint64());
      }
    }
    std::sort(
        _ps_server_list.begin(),
        _ps_server_list.end(),
        [](const PSHost &h1, const PSHost &h2) { return h1.rank < h2.rank; });
    return 0;
  }

  virtual int32_t SetPsServers(const std::vector<std::string> *host_sign_list,
                               int node_num) {
    _ps_server_list.clear();
    _ps_server_sign_set.clear();
    for (int i = 0; i < node_num; ++i) {
      if (host_sign_list->at(i) != "") {
        PSHost host;
        host.ParseFromString(host_sign_list->at(i));
        _ps_server_list.push_back(host);
        _ps_server_sign_set.insert(host.rank);
      }
    }
    std::sort(
        _ps_server_list.begin(),
        _ps_server_list.end(),
        [](const PSHost &h1, const PSHost &h2) { return h1.rank < h2.rank; });
    return 0;
  }

  virtual int32_t SetPsClients(uint64_t *host_sign_list, int node_num) {
    _ps_client_list.clear();
    _ps_client_sign_set.clear();
    for (int i = 0; i < node_num; ++i) {
      if (host_sign_list[i] > 0) {
        PSHost host;
        host.ParseFromUint64(host_sign_list[i]);
        _ps_client_list.push_back(host);
        _ps_client_sign_set.insert(host.SerializeToUint64());
      }
    }
    std::sort(
        _ps_client_list.begin(),
        _ps_client_list.end(),
        [](const PSHost &h1, const PSHost &h2) { return h1.rank < h2.rank; });
    return 0;
  }

  virtual int32_t SetPsClients(const std::vector<std::string> *host_sign_list,
                               int node_num) {
    _ps_client_list.clear();
    _ps_client_sign_set.clear();
    for (int i = 0; i < node_num; ++i) {
      if (host_sign_list->at(i) != "") {
        PSHost host;
        host.ParseFromString(host_sign_list->at(i));
        _ps_client_list.push_back(host);
        _ps_client_sign_set.insert(host.rank);
      }
    }
    std::sort(
        _ps_client_list.begin(),
        _ps_client_list.end(),
        [](const PSHost &h1, const PSHost &h2) { return h1.rank < h2.rank; });
    VLOG(1) << "env.set_ps_clients done\n";
    return 0;
  }

  virtual void SetCoordinators(const std::vector<std::string> *host_sign_list,
                               size_t node_num) {
    _coordinator_list.clear();
    _coordinator_sign_set.clear();
    for (size_t i = 0; i < node_num; ++i) {
      if (host_sign_list->at(i) != "") {
        PSHost host;
        host.ParseFromString(host_sign_list->at(i));
        _coordinator_list.push_back(host);
        _coordinator_sign_set.insert(host.rank);
        VLOG(0) << "fl-ps > coordinator info in env: " << host.ToString();
      }
    }
    return;
  }

  virtual uint64_t GetLocalHostSign() {
    if (_ps_client_list.size() > 0) {
      return _ps_client_list[0].SerializeToUint64();
    } else {
      return 0;
    }
  }
};

}  // namespace distributed
}  // namespace paddle
