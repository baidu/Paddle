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

#pragma once

#include <ucc/api/ucc.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/distributed/types.h"

namespace phi {
namespace distributed {

#define UCC_CHECK(cmd, error_msg)                                            \
  do {                                                                       \
    ucc_status_t r = cmd;                                                    \
    if (r != UCC_OK) {                                                       \
      PADDLE_THROW(common::errors::External(                                 \
          "Failed, UCC error %s:%d '%s'\n", __FILE__, __LINE__, error_msg)); \
    }                                                                        \
  } while (0)

// Macro and throw on a non-successful UCC return value and free its request.
#define UCC_CHECK_REQUEST(request, cmd, error_msg)                           \
  do {                                                                       \
    ucc_status_t r = cmd;                                                    \
    if (r != UCC_OK) {                                                       \
      if (request != nullptr) {                                              \
        ucc_collective_finalize(request);                                    \
      }                                                                      \
      PADDLE_THROW(common::errors::External(                                 \
          "Failed, UCC error %s:%d '%s'\n", __FILE__, __LINE__, error_msg)); \
    }                                                                        \
  } while (0)

#define PADDLE_CHECK(expr)                                                \
  do {                                                                    \
    if (!(expr)) {                                                        \
      PADDLE_THROW(common::errors::Unavailable(                           \
          "Failed, expr error %s:%d '%s'\n", __FILE__, __LINE__, #expr)); \
    }                                                                     \
  } while (0)

ucc_reduction_op_t ToUCCRedType(ReduceOp reduction);

ucc_memory_type_t ToUCCMemType(phi::AllocationType type);

std::string UCCDTypeToString(ucc_datatype_t dtype);

std::string UCCRedTypeToString(ucc_reduction_op_t op);

// trim: remove spaces before and after the string view
// implementation borrowed from https://stackoverflow.com/a/17976541
inline std::string Trim(std::string s) {
  auto wsfront = std::find_if_not(
      s.begin(), s.end(), [](int c) { return std::isspace(c); });
  auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c) {
                  return std::isspace(c);
                }).base();
  return (wsback <= wsfront ? ""
                            : s.substr(wsfront - s.begin(), wsback - wsfront));
}

inline std::string ToLower(std::string s) {
  std::string result;
  result.reserve(s.size());
  for (auto c : s) {
    result.push_back(std::tolower(c));
  }
  return result;
}

inline std::vector<std::string> ParseList(std::string list) {
  std::vector<std::string> result;
  list = ToLower(Trim(list));
  while (!list.empty()) {
    const auto end_pos = list.find_first_of(',');
    const auto token = Trim(list.substr(0, end_pos));
    result.push_back(std::string(token));
    list = (end_pos != std::string::npos) ? list.substr(end_pos + 1) : "";
  }
  return result;
}

struct PaddleUCCOobCollInfo {
  std::shared_ptr<Store> store;
  uint32_t comm_id;
  int rank;
  int size;
  void* rbuf;
  size_t msglen;
  std::string GetKey(std::string key) { return std::to_string(comm_id) + key; }
};

ucc_status_t OobAllgather(
    void* sbuf, void* rbuf, size_t msglen, void* coll_info, void** req);

ucc_status_t OobAllgatherTest(void* req);

ucc_status_t OobAllgatherFree(void* req);

class UCCCommBase {
 public:
  ucc_lib_h lib{nullptr};
  ucc_context_h context{nullptr};

  explicit UCCCommBase(std::shared_ptr<PaddleUCCOobCollInfo> oob);
  ~UCCCommBase();
  virtual void Progress();
  virtual void FreeRequest(ucc_coll_req_h request);
};

}  // namespace distributed
}  // namespace phi
