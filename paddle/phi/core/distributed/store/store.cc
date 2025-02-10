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

#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/enforce.h"

namespace phi::distributed {

int64_t Store::add(const std::string& key, int64_t value) {
  PADDLE_THROW(
      errors::InvalidArgument("Implement the add method in the subclass."));
}

std::vector<uint8_t> Store::get(const std::string& key) {
  PADDLE_THROW(
      errors::InvalidArgument("Implement the get method in the subclass."));
}

bool Store::check(const std::string& key) {
  PADDLE_THROW(
      errors::InvalidArgument("Implement the get method in the subclass."));
}

void Store::wait(const std::string& key) {
  PADDLE_THROW(
      errors::InvalidArgument("Implement the wait method in the subclass."));
}

void Store::set(const std::string& key, const std::vector<uint8_t>& value) {
  PADDLE_THROW(
      errors::InvalidArgument("Implement the set method in the subclass."));
}

bool Store::deleteKey(const std::string& key) {
  PADDLE_THROW(errors::InvalidArgument(
      "Implement the deleteKey method in the subclass."));
}

}  // namespace phi::distributed
