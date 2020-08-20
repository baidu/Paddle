//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"

namespace paddle {
namespace imperative {

struct GlooParallelStrategy {
  int rank{0};
  int rank_num{1};
  std::string iface;
  std::string prefix;
  int init_seconds{9999999};
  int run_seconds{9999999};
  std::string path;
  std::string fs_name;
  std::string fs_ugi;
};

#if defined(PADDLE_WITH_GLOO)
class GlooParallelContext {
 public:
  explicit GlooParallelContext(const GlooParallelStrategy& strategy)
      : strategy_(strategy) {}

  virtual ~GlooParallelContext() {}

  virtual void Init();

 protected:
  GlooParallelStrategy strategy_;
};
#endif
}  //  namespace imperative
}  //  namespace paddle
