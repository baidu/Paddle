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

#include "paddle/cinn/optim/transform_polyfor_to_for.h"

#include <cmath>
#include <vector>

#include "paddle/cinn/common/arithmetic.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/ir_simplify.h"

namespace cinn {
namespace optim {

namespace {

Expr PlusOneWithMinMax(Expr expr) {
  auto* min_n = expr.As<ir::Min>();
  auto* max_n = expr.As<ir::Max>();

  if (min_n) {
    min_n->a() = min_n->a() + 1;
    min_n->b() = min_n->b() + 1;
    Simplify(&min_n->a());
    Simplify(&min_n->b());
    return expr;
  } else if (max_n) {
    max_n->a() = max_n->a() + 1;
    max_n->b() = max_n->b() + 1;
    Simplify(&max_n->a());
    Simplify(&max_n->b());
    return expr;
  }
  return expr + 1;
}

struct PolyForWithSimpleConditionToForMutator : public ir::IRMutator<Expr*> {
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    auto* ge_n = node->condition.As<ir::GE>();
    auto* gt_n = node->condition.As<ir::GT>();
    if (ge_n) {
      node->condition = (ge_n->a() * -1) <= (ge_n->b() * -1);
    }
    if (gt_n) {
      node->condition = (ge_n->a() * -1) < (ge_n->b() * -1);
    }

    auto* lt_n = node->condition.As<ir::LT>();
    auto* le_n = node->condition.As<ir::LE>();

    if (lt_n) {
      if (lt_n->b() != cinn::common::make_const(0)) {
        node->condition = lt_n->a() - lt_n->b() < 0;
      }
    }
    if (le_n) {
      if (le_n->b() != cinn::common::make_const(0)) {
        node->condition = le_n->a() - le_n->b() <= 0;
      }
    }

    lt_n = node->condition.As<ir::LT>();
    le_n = node->condition.As<ir::LE>();
    if (!(lt_n || le_n)) return;

    // check the lhs is the iterator
    bool can_extract_extent =
        (lt_n && lt_n->a().as_var() &&
         lt_n->a().as_var()->name == op->iterator->name) ||
        (le_n && le_n->a().as_var() &&
         le_n->a().as_var()->name == op->iterator->name);

    if (!can_extract_extent) {
      if (node->condition.As<ir::LE>()) {
        auto le = node->condition.As<ir::LE>();

        PADDLE_ENFORCE_NOT_NULL(
            le->a().As<ir::Sub>(),
            ::common::errors::InvalidArgument("The value of le is incorrect."
                                              "Expected value is 0"));
        PADDLE_ENFORCE_EQ(le->b().As<ir::IntImm>()->value,
                          0UL,
                          ::common::errors::InvalidArgument(
                              "The value of le is incorrect."
                              "Expected value is 0, but receive %d.",
                              le->b().As<ir::IntImm>()->value));
        auto sub = le->a().As<ir::Sub>();
        node->condition = ir::LE::Make(sub->a(), sub->b());
      } else if (node->condition.As<ir::LT>()) {
        auto lt = node->condition.As<ir::LT>();
        PADDLE_ENFORCE_NOT_NULL(
            lt->a().As<ir::Sub>(),
            ::common::errors::InvalidArgument("The value of lt is incorrect."
                                              "Expected value is 0"));
        PADDLE_ENFORCE_EQ(lt->b().As<ir::IntImm>()->value,
                          0UL,
                          ::common::errors::InvalidArgument(
                              "The value of lt is incorrect."
                              "Expected value is 0, but receive %d.",
                              lt->b().As<ir::IntImm>()->value));
        auto sub = lt->a().As<ir::Sub>();
        node->condition = ir::LT::Make(sub->a(), sub->b());
      } else {
        PADDLE_THROW(::common::errors::InvalidArgument("Unknown Type!"));
      }

      lt_n = node->condition.As<ir::LT>();
      le_n = node->condition.As<ir::LE>();
      if (!(lt_n || le_n)) return;
    }

    Expr lhs = lt_n ? lt_n->a() : le_n->a();
    Expr rhs = lt_n ? lt_n->b() : PlusOneWithMinMax(le_n->b());
    rhs = cinn::common::AutoSimplify(rhs);

    if (op->is_vectorized())
      PADDLE_ENFORCE_EQ(
          op->vectorize_info().valid(),
          true,
          ::common::errors::InvalidArgument(
              "The value of op->vectorize_info().valid() is incorrect."
              "Expected value is true"));

    Expr new_for = ir::For::Make(op->iterator,
                                 op->init,
                                 rhs,
                                 op->for_type(),
                                 op->device_api,
                                 op->body,
                                 op->vectorize_info());
    *expr = new_for;

    Visit(&new_for.As<ir::For>()->body);
  }
};

}  // namespace

void TransformPolyForToFor(Expr* expr, bool auto_separate) {
  PolyForWithSimpleConditionToForMutator()(expr);
}

}  // namespace optim
}  // namespace cinn
