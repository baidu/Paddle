// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/utils/ir_copy.h"

#include <gtest/gtest.h>

#include "paddle/cinn/ir/ir_printer.h"

namespace cinn {
namespace ir {
namespace ir_utils {

TEST(IrCopy, basic) {
  Expr a(1.f);
  auto aa = IRCopy(a);
  LOG(INFO) << "aa " << aa;
}
}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
