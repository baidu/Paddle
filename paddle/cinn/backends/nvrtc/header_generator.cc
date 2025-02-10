// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/nvrtc/header_generator.h"

#include "glog/logging.h"
#include "jitify.hpp"  // NOLINT
#include "paddle/common/enforce.h"
namespace cinn {
namespace backends {
namespace nvrtc {

HeaderGeneratorBase& JitSafeHeaderGenerator::GetInstance() {
  static JitSafeHeaderGenerator instance;
  return instance;
}

const size_t JitSafeHeaderGenerator::size() const {
  PADDLE_ENFORCE_EQ(include_names_.size(),
                    headers_.size(),
                    ::common::errors::InvalidArgument(
                        "Internal error in size of header files."));
  return include_names_.size();
}

JitSafeHeaderGenerator::JitSafeHeaderGenerator() {
  const auto& headers_map = ::jitify::detail::get_jitsafe_headers_map();
  for (auto& pair : headers_map) {
    if (pair.first == "cstdint" || pair.first == "stdint.h" ||
        pair.first == "climits" || pair.first == "stddef.h" ||
        pair.first == "stdlib.h") {
      include_names_.emplace_back(pair.first.data());
      headers_.emplace_back(pair.second.data());
    }
  }
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn
