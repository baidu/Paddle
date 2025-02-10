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
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace phi {
namespace distributed {

class Store {
 public:
  Store() : _timeout(900) {}
  explicit Store(const int timeout) : _timeout(timeout) {}
  virtual ~Store() = default;

  virtual int64_t add(const std::string& key, int64_t value);
  virtual std::vector<uint8_t> get(const std::string& key);
  virtual bool check(const std::string& key);
  virtual void wait(const std::string& key);
  virtual void set(const std::string& key, const std::vector<uint8_t>& value);
  virtual bool deleteKey(const std::string& key);

  virtual int timeout() { return _timeout; }

 protected:
  int _timeout;
};

}  // namespace distributed
}  // namespace phi
