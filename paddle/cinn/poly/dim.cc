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

#include "paddle/cinn/poly/dim.h"

#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/optim/ir_simplify_pass.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace poly {

std::string Dim::range_repr() const {
  return utils::StringFormat("%s <= %s <= %s",
                             utils::GetStreamCnt(lower_bound).c_str(),
                             id.c_str(),
                             utils::GetStreamCnt(upper_bound).c_str());
}

Dim::Dim(std::string id, ir::Expr lower_bound, ir::Expr upper_bound)
    : id(std::move(id)), lower_bound(lower_bound), upper_bound(upper_bound) {
  optim::Simplify(&this->lower_bound, optim::SimplifyType::kExpr);
  optim::Simplify(&this->upper_bound, optim::SimplifyType::kExpr);
}

}  // namespace poly
}  // namespace cinn
