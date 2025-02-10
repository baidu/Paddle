// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/stmt.h"
#include "paddle/cinn/ir/tensor.h"

namespace cinn {
namespace ast_gen_ius {

class AstGen {
 public:
  static ir::stmt::StmtRef Build(const ir::Tensor& tensor,
                                 TensorGroup* tensor_group);
};

}  // namespace ast_gen_ius
}  // namespace cinn
