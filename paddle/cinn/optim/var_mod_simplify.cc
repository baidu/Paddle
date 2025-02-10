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

#include "paddle/cinn/optim/var_mod_simplify.h"

#include <absl/container/flat_hash_map.h>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"

namespace cinn::optim {

namespace {
using namespace ir;  // NOLINT

struct ReplaceModWithDivMutator : public ir::IRMutator<> {
  void operator()(Expr* x) { ir::IRMutator<>::Visit(x, x); }

  void Visit(const Mod* op, Expr* expr) override {
    auto* node = expr->As<ir::Mod>();
    auto a = node->operand(0);
    auto b = node->operand(1);
    *expr = ir::Div::Make(a, b);
    *expr = ir::Mul::Make(b, *expr);
    *expr = ir::Sub::Make(a, *expr);
  }
};

struct ReplaceDivWithVarMutator : public ir::IRMutator<> {
  absl::flat_hash_map<std::string, Expr> div_var_map_;
  void operator()(Expr* x) { ir::IRMutator<>::Visit(x, x); }

  void Visit(const Div* op, Expr* expr) override {
    auto* node = expr->As<ir::Div>();

    auto a = node->operand(0);
    auto b = node->operand(1);
    // only deal with var/int
    if (a.is_var() && b.is_constant()) {
      auto a_var = a.As<_Var_>();
      auto b_int = b.As<IntImm>();
      PADDLE_ENFORCE_NOT_NULL(a_var,
                              ::common::errors::InvalidArgument(
                                  "The node->operand(0) should be var"));
      PADDLE_ENFORCE_NOT_NULL(b_int,
                              ::common::errors::InvalidArgument(
                                  "The node->operand(1) should be int"));
      std::string var_name = a_var->name + "/" + std::to_string(b_int->value);
      div_var_map_[var_name] = ir::Div::Make(a, b);
      *expr = Var(var_name);
    }
  }
};

struct ReplaceVarWithDivMutator : public ir::IRMutator<> {
  absl::flat_hash_map<std::string, Expr> div_var_map_;
  void operator()(Expr* x,
                  const absl::flat_hash_map<std::string, Expr>& div_var_map) {
    div_var_map_ = div_var_map;
    ir::IRMutator<>::Visit(x, x);
  }

  void Visit(const _Var_* op, Expr* expr) override {
    auto* node = expr->As<_Var_>();
    PADDLE_ENFORCE_NOT_NULL(node,
                            ::common::errors::InvalidArgument(
                                "Sorry, but the node expr is nullptr"));
    if (div_var_map_.count(node->name)) {
      *expr = div_var_map_[node->name];
    }
  }
};

}  // namespace

void VarModSimplify(Expr* e) {
  *e = cinn::common::AutoSimplify(*e);
  ReplaceModWithDivMutator()(e);
  ReplaceDivWithVarMutator mutator;
  mutator(e);
  *e = cinn::common::AutoSimplify(*e);
  auto div_var_map = mutator.div_var_map_;
  ReplaceVarWithDivMutator()(e, mutator.div_var_map_);
}

}  // namespace cinn::optim
