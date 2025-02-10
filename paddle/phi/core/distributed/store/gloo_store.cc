// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/store/gloo_store.h"

<<<<<<< HEAD
#include <utility>

namespace phi {
namespace distributed {
=======
namespace phi::distributed {
>>>>>>> 337f0d225e59843df33c4ca189ae312b8c36a4f3

GlooStore::GlooStore(std::shared_ptr<phi::distributed::Store>  store)
    : store_(std::move(store)) {}

std::vector<char> GlooStore::get(const std::string& key) {
  auto value = store_->get(key);
  return std::vector<char>(value.begin(), value.end());
}

void GlooStore::wait(const std::vector<std::string>& keys) {
  for (auto& key : keys) {
    store_->wait(key);
  }
}

void GlooStore::set(const std::string& key, const std::vector<char>& value) {
  std::vector<uint8_t> tmp(value.begin(), value.end());
  store_->set(key, tmp);
}

void GlooStore::wait(const std::vector<std::string>& keys,
                     const std::chrono::milliseconds& timeout) {
  for (auto& key : keys) {
    store_->wait(key);
  }
}

}  // namespace phi::distributed
