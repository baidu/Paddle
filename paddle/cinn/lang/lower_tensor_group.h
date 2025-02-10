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
#include <absl/container/flat_hash_map.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/optim/buffer_assign.h"
#include "paddle/cinn/optim/compute_inline_expand.h"
#include "paddle/cinn/optim/fold_cinn_call_arguments.h"
#include "paddle/cinn/optim/optimize.h"
#include "paddle/cinn/optim/replace_call_with_expr.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"
#include "paddle/cinn/poly/ast_gen.h"

namespace cinn {
namespace lang {
namespace detail {

class LowerTensorGroup {
 public:
  LowerTensorGroup(const std::string& fn_name,
                   const std::vector<ir::Tensor>& tensor_args,
                   const std::vector<ir::Var>& scalar_args,
                   ast_gen_ius::TensorGroup* tensor_group,
                   const std::vector<ir::Tensor>& temp_tensor_args = {},
                   const Target& target = cinn::common::DefaultHostTarget());

  std::vector<ir::LoweredFunc> operator()();

  std::vector<ir::stmt::BlockRef> GenerateFunctionBody(
      ast_gen_ius::TensorGroup* tensor_group);

  std::vector<ir::Argument> GenerateFunctionArgumentList(
      const ir::stmt::BlockRef& fn_body);

 private:
  const std::string& fn_name_;
  const std::vector<ir::Tensor>& tensor_args_;
  const std::vector<Var>& scalar_args_;
  std::vector<ir::Tensor> temp_tensor_args_;
  ast_gen_ius::TensorGroup* tensor_group_;
  Target target_;
};

}  // namespace detail
}  // namespace lang
}  // namespace cinn
