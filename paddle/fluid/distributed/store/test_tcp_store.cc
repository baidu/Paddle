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

#include <unistd.h>

#include <iostream>
#include <unordered_map>

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/store/tcp_store.h"
#include "paddle/fluid/distributed/store/tcp_utils.h"

namespace paddle {
namespace distributed {

TEST(MasterDaemon, init) {
  int port = 6170;
  int socket = tcputils::tcp_listen("", std::to_string(port), AF_INET);
  auto d = detail::MasterDaemon::start(socket, 1, 100);
  printf("started to sleep 1\n");
  usleep(2 * 1000 * 1000);
  printf("end to reset\n");

  d.reset();

  // printf("started to sleep 5\n");
  // usleep(5*1000*1000);
  // printf("end to exit\n");
}

};  // namespace distributed
};  // namespace paddle
