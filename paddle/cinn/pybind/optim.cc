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

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/ir_simplify_pass.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/pybind/bind_utils.h"
#include "paddle/cinn/utils/string.h"

namespace py = pybind11;

namespace cinn::pybind {

using optim::Simplify;

namespace {

void BindSimplify(py::module* m) {
  m->def(
      "simplify",
      [](const Expr& expr) -> Expr {
        auto copied = ir::ir_utils::IRCopy(expr);
        Simplify(&copied, optim::SimplifyType::kBlock);
        return copied;
      },
      py::arg("expr"));

  m->def("ir_copy",
         py::overload_cast<const Expr&, bool>(&ir::ir_utils::IRCopy),
         py::arg("x"),
         py::arg("copy_buffer_node") = true);
}

}  // namespace

void BindOptim(py::module* m) { BindSimplify(m); }

}  // namespace cinn::pybind
