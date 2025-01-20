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

#include "paddle/cinn/optim/ir_simplify_pass.h"

#include <absl/container/flat_hash_map.h>
#include <ginac/ginac.h>
#include <glog/logging.h>

#include <map>
#include <string>

#include "paddle/cinn/common/arithmetic.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/pass/pass.h"
#include "paddle/cinn/pass/pass_manager.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT
using cinn::common::bfloat16;
using cinn::common::ExprToGinacConverter;
using cinn::common::float16;
using utils::GetStreamCnt;
using utils::Replace;

namespace {

//! Simplify the expression but Load.
class SimplifyNoPureMathMutator : public ir::IRMutator<ir::Expr*> {
 public:
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

 private:
  using ir::IRMutator<>::Visit;

#define __(op__)                                    \
  void Visit(const op__* op, Expr* expr) override { \
    *expr = ArithSimplify(*expr);                   \
  }

  __(Add)
  __(Mul)
  __(Sub)
  __(Div)
  __(Min)
  __(Max)
#undef __
};

class SimplifyLoadMutator : public ir::IRMutator<ir::Expr*> {
 public:
  void operator()(ir::Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

 private:
  void Visit(const Load* expr, Expr* op) override {
    auto* node = op->As<Load>();
    for (auto& idx : node->indices) {
      if (cinn::common::IsPureMath(idx)) {
        idx = ArithSimplify(idx);
      } else {
        SimplifyNoPureMathMutator()(&idx);
      }
    }
  }
};

class SimplifyStoreMutator {
 public:
  void operator()(ir::stmt::BlockRef block) {
    for (auto& stmt : block->stmts()) {
      if (stmt->stmt_type() == ir::StmtNodeTy::Store) {
        VisitStmt(stmt.as<ir::stmt::Store>());
      }
    }
  }

 private:
  void VisitStmt(ir::stmt::Store stmt) {
    std::vector<Expr> new_indices = stmt->indices();
    for (ir::Expr& index : new_indices) {
      if (cinn::common::IsPureMath(index)) {
        index = ArithSimplify(index);
      } else {
        SimplifyNoPureMathMutator()(&index);
      }
    }
    stmt->set_indices(new_indices);
  }
};

class SimplifyRampMutator : public ir::IRMutator<Expr*> {
 public:
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

 private:
  void Visit(const Ramp* op, Expr* expr) override {
    auto* node = expr->As<ir::Ramp>();

    PADDLE_ENFORCE_EQ(
        cinn::common::IsPureMath(node->base),
        true,
        ::common::errors::InvalidArgument("node->base is not a pure math!"));
    PADDLE_ENFORCE_EQ(
        cinn::common::IsPureMath(node->stride),
        true,
        ::common::errors::InvalidArgument("node->stride is not a pure math!"));
    node->base = ArithSimplify(node->base);
    node->stride = ArithSimplify(node->stride);
  }
  // ramp + ramp
  void Visit(const Add* op, Expr* expr) override {
    auto* node = expr->As<ir::Add>();
    Expr a = node->a();
    Expr b = node->b();
    auto a_ramp = a.As<ir::Ramp>();
    auto b_ramp = b.As<ir::Ramp>();

    if (a_ramp && b_ramp && a_ramp->lanes == b_ramp->lanes) {
      Expr base_add = cinn::common::AutoSimplify(a_ramp->base + b_ramp->base);
      Expr stride_add =
          cinn::common::AutoSimplify(a_ramp->stride + b_ramp->stride);
      *expr = ir::Ramp::Make(base_add, stride_add, a_ramp->lanes);
    }
  }
};

class SimplifyIfThenElseMutator {
 public:
  void operator()(ir::stmt::BlockRef block) { VisitBlock(block); }

 private:
  void VisitBlock(ir::stmt::BlockRef block) {
    std::unordered_set<int> empty_stmt_id;
    std::vector<ir::stmt::StmtRef> stmts = block->stmts();
    for (int i = 0; i < stmts.size(); i++) {
      if (stmts[i].isa<ir::stmt::IfThenElse>())
        if (IsEmptyIf(stmts[i].as<ir::stmt::IfThenElse>()))
          empty_stmt_id.insert(i);
    }

    std::vector<ir::stmt::StmtRef> new_stmts;
    for (int i = 0; i < stmts.size(); i++) {
      if (!empty_stmt_id.count(i)) new_stmts.push_back(stmts[i]);
    }
    block->set_stmts(new_stmts);
  }

  bool IsEmptyIf(ir::stmt::IfThenElse stmt) {
    const Expr& condition = stmt->condition();
    stmt->set_condition(ArithSimplify(condition));

    auto* condition_int = stmt->condition().As<ir::IntImm>();
    auto* condition_uint = stmt->condition().As<ir::UIntImm>();

    // not deterministic
    if (!condition_int && !condition_uint) {
      VisitBlock(stmt->true_case());
      if (stmt->false_case().defined()) {
        VisitBlock(stmt->false_case());
      }
      return false;
    }

    bool value = condition_int ? condition_int->value : condition_uint->value;
    if (value) {
      VisitBlock(stmt->true_case());
      return false;
    } else if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
      return false;
    } else {
      return true;
    }
  }
};

class SimplifySelectMutator : public ir::IRMutator<> {
 public:
  void operator()(Expr* x) { ir::IRMutator<>::Visit(x, x); }

 private:
  void Visit(const Select* op, Expr* expr) override {
    auto* node = expr->As<ir::Select>();

    auto* condition_int = node->condition.As<ir::IntImm>();
    auto* condition_uint = node->condition.As<ir::UIntImm>();

    // not deterministic
    if (!condition_int && !condition_uint) {
      operator()(&node->true_value);
      operator()(&node->false_value);
      return;
    }

    bool value = condition_int ? condition_int->value : condition_uint->value;
    if (value) {
      *expr = op->true_value;
      ir::IRMutator<>::Visit(expr, expr);
    } else {
      *expr = op->false_value;
      ir::IRMutator<>::Visit(expr, expr);
    }
  }
};

struct SimplifyLogicalMutator : public ir::ExprMutator<> {
 public:
  void operator()(Expr* expr) { ir::ExprMutator<>::Visit(expr, expr); }

 private:
#define DEFINE_VISIT_CMP_OP(OpType, Method)                         \
  void Visit(const ir::OpType* op, Expr* expr) override {           \
    VLOG(7) << "Begin Visit Cmp op: " << *expr;                     \
    auto* node = expr->As<ir::OpType>();                            \
    ir::ExprMutator<>::Visit(&node->a(), &node->a());               \
    ir::ExprMutator<>::Visit(&node->b(), &node->b());               \
    if (node->a().is_constant() && node->b().is_constant())         \
      if (node->a().get_constant() Method node->b().get_constant()) \
        *expr = Expr(true);                                         \
    VLOG(7) << "End Visit Cmp op: " << *expr;                       \
  }
  DEFINE_VISIT_CMP_OP(LE, <=)
  DEFINE_VISIT_CMP_OP(LT, <)
  DEFINE_VISIT_CMP_OP(GE, >=)
  DEFINE_VISIT_CMP_OP(GT, >)
  DEFINE_VISIT_CMP_OP(EQ, ==)
  DEFINE_VISIT_CMP_OP(NE, !=)

#undef DEFINE_VISIT_CMP_OP

  void Visit(const ir::And* op, Expr* expr) override {
    VLOG(7) << "Begin Visit And op: " << *expr;
    auto* node = expr->As<ir::And>();
    ir::ExprMutator<>::Visit(&node->a(), &node->a());
    if (common::IsZero(node->a())) {
      *expr = Expr(false);
      VLOG(7) << "End Visit And op: " << *expr;
      return;
    }
    ir::ExprMutator<>::Visit(&node->b(), &node->b());
    if (common::IsZero(node->b())) {
      VLOG(7) << "End Visit And op: " << *expr;
      *expr = Expr(false);
      return;
    }
    if (common::IsOne(node->a()) && common::IsOne(node->b()))
      *expr = Expr(true);
    VLOG(7) << "End Visit And op: " << *expr;
  }

  void Visit(const ir::Or* op, Expr* expr) override {
    VLOG(7) << "Begin Visit Or op: " << *expr;
    auto* node = expr->As<ir::Or>();
    ir::ExprMutator<>::Visit(&node->a(), &node->a());
    if (common::IsOne(node->a())) {
      *expr = Expr(true);
      VLOG(7) << "End visit Or op: " << *expr;
      return;
    }
    ir::ExprMutator<>::Visit(&node->b(), &node->b());
    if (common::IsOne(node->b())) {
      *expr = Expr(true);
      VLOG(7) << "End visit Or op: " << *expr;
      return;
    }
    if (common::IsZero(node->a()) && common::IsZero(node->b()))
      *expr = Expr(false);
    VLOG(7) << "End visit Or op: " << *expr;
  }

  void Visit(const ir::Not* op, Expr* expr) override {
    VLOG(7) << "Begin Visit Not op: " << *expr;
    auto* node = expr->As<ir::Not>();
    auto v = node->v();
    ir::ExprMutator<>::Visit(&v, &v);
    switch (v.node_type()) {
      case ir::IrNodeTy::IntImm:
      case ir::IrNodeTy::UIntImm:
        *expr = common::IsZero(v) ? Expr(true) : Expr(false);
        return;
      case ir::IrNodeTy::Not:
        *expr = v.As<ir::Not>()->v();
        return;
      case ir::IrNodeTy::LE:
        *expr = ir::GT::Make(v->operand(0), v->operand(1));
        return;
      case ir::IrNodeTy::LT:
        *expr = ir::GE::Make(v->operand(0), v->operand(1));
        return;
      case ir::IrNodeTy::GE:
        *expr = ir::LT::Make(v->operand(0), v->operand(1));
        return;
      case ir::IrNodeTy::GT:
        *expr = ir::LE::Make(v->operand(0), v->operand(1));
        return;
      default:
        VLOG(7) << "End Visit Not op: " << *expr;
        return;
    }
    VLOG(7) << "End Visit Not op: " << *expr;
  }
};

class SimplifyLogicalPass : public ExprPass {
 public:
  SimplifyLogicalPass() : ExprPass("simplify_logical_pass") {}

  LogicalResult Run(ir::Expr* expr) override {
    SimplifyLogicalMutator mutator;
    mutator(expr);
    return LogicalResult::success();
  }
};
std::unique_ptr<ExprPass> CreateSimplifyLogicalPass() {
  return std::make_unique<SimplifyLogicalPass>();
}

struct ReplaceFracWithDivMutator : public ir::IRMutator<> {
 public:
  void operator()(Expr* x) { ir::IRMutator<>::Visit(x, x); }

 private:
  void Visit(const FracOp* op, Expr* expr) override {
    auto* node = expr->As<ir::FracOp>();

    ir::IRMutator<>::Visit(&node->operand(0), &node->operand(0));
    ir::IRMutator<>::Visit(&node->operand(1), &node->operand(1));

    *expr = ir::Div::Make(node->operand(0), node->operand(1));
  }
};

class SimplifyFracWithDivPass : public ExprPass {
 public:
  SimplifyFracWithDivPass() : ExprPass("simplify_frac_with_div_pass") {}

  LogicalResult Run(ir::Expr* expr) override {
    ReplaceFracWithDivMutator mutator;
    mutator(expr);
    return LogicalResult::success();
  }
};
std::unique_ptr<ExprPass> CreateSimplifyFracWithDivPass() {
  return std::make_unique<SimplifyFracWithDivPass>();
}

class SimplifyForLoopsMutator : public ir::IRMutator<ir::Expr*>,
                                public ir::stmt::StmtMutator<bool, bool> {
 public:
  void operator()(ir::Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void operator()(ir::stmt::BlockRef block) { VisitBlock(block); }

 private:
  bool VisitBlock(ir::stmt::BlockRef block) override {
    std::vector<ir::stmt::StmtRef> stmts = block->stmts();
    for (auto i = stmts.size() - 1; i >= 0; i--) {
      if (stmts[i].isa<ir::stmt::For>()) {
        if (VisitStmt(stmts[i].as<ir::stmt::For>())) {
          const std::vector<ir::stmt::StmtRef>& inner_stmts =
              stmts[i].as<ir::stmt::For>()->body()->stmts();
          stmts.insert(
              stmts.begin() + i, inner_stmts.begin(), inner_stmts.end());
          stmts.erase(stmts.begin() + i + inner_stmts.size());
        }
      } else {
        ir::stmt::StmtMutator<bool, bool>::VisitStmt(stmts[i]);
      }
    }
    block->set_stmts(stmts);
  }

  bool VisitStmt(ir::stmt::For stmt) override {
    Expr min = stmt->min();
    Expr extent = stmt->extent();
    operator()(&min);
    operator()(&extent);
    stmt->set_min(min);
    stmt->set_extent(extent);
    const IntImm* min_i = stmt->min().As<IntImm>();
    const IntImm* extent_i = stmt->extent().As<IntImm>();
    if (min_i && extent_i && (extent_i->value - min_i->value) == 1) {
      VLOG(6) << "Simplify current For Loop";
      std::string var_name = stmt->loop_var()->name;
      var_mins.emplace(var_name, stmt->min());
      VisitBlock(stmt->body());
      var_mins.erase(var_name);
      return true;
    } else {
      VisitBlock(stmt->body());
      return false;
    }
  }

  bool VisitStmt(ir::stmt::IfThenElse stmt) override {
    Expr condition = stmt->condition();
    operator()(&condition);
    stmt->set_condition(condition);

    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  bool VisitStmt(ir::stmt::Schedule stmt) override {
    std::vector<Expr> iter_values = stmt->iter_values();
    std::vector<Expr> read_buffers = stmt->read_buffers();
    std::vector<Expr> write_buffers = stmt->write_buffers();

    for (Expr& iter_value : iter_values) operator()(&iter_value);
    for (Expr& read_buffer : read_buffers) operator()(&read_buffer);
    for (Expr& write_buffer : write_buffers) operator()(&write_buffer);

    stmt->set_iter_values(iter_values);
    stmt->set_read_buffers(read_buffers);
    stmt->set_write_buffers(write_buffers);
  }

  bool VisitStmt(ir::stmt::Let stmt) override {
    Expr value = stmt->body();
    operator()(&value);
    stmt->set_body(value);
  }

  bool VisitStmt(ir::stmt::Store stmt) override {
    Expr value = stmt->value();
    operator()(&value);
    stmt->set_value(value);

    std::vector<Expr> indices = stmt->indices();
    for (Expr& index : indices) {
      operator()(&index);
    }
    stmt->set_indices(indices);
  }

  bool VisitStmt(ir::stmt::Alloc stmt) override {
    Expr condition = stmt->condition();
    operator()(&condition);
    stmt->set_condition(condition);
  }

  bool VisitStmt(ir::stmt::Free stmt) override {
    Expr destination = stmt->destination();
    operator()(&destination);
    stmt->set_destination(destination);
  }

  bool VisitStmt(ir::stmt::Evaluate stmt) override {}

  void Visit(const _Var_* op, Expr* expr) override {
    auto* node = expr->As<ir::_Var_>();

    if (var_mins.count(node->name)) {
      *expr = var_mins.at(node->name);
    }
  }

  absl::flat_hash_map<std::string, Expr> var_mins;
};

template <typename CastType, typename T>
CastType NormCastValue(T value) {
  if (type_of<CastType>().is_uint() || type_of<T>().is_uint()) {
    // not support uint
    return static_cast<CastType>(value);
  }

  if (std::isinf(value)) {
    if (CastType(value) == -std::numeric_limits<CastType>::infinity()) {
      return -std::numeric_limits<CastType>::infinity();
    }
    return std::numeric_limits<CastType>::infinity();
  } else if (std::isnan(value)) {
    return std::numeric_limits<CastType>::signaling_NaN();
  } else if (value >= static_cast<T>(std::numeric_limits<CastType>::max())) {
    return std::numeric_limits<CastType>::max();
  } else if (value <= static_cast<T>(std::numeric_limits<CastType>::lowest())) {
    return std::numeric_limits<CastType>::lowest();
  }
  return static_cast<CastType>(value);
}

class SimplifyCastMutator : public ir::IRMutator<> {
 public:
  void operator()(Expr* x) { ir::IRMutator<>::Visit(x, x); }

 private:
  void Visit(const ir::Cast* op, Expr* expr) override {
    auto* node = expr->As<ir::Cast>();

    ir::IRMutator<ir::Expr*>::Visit(&node->v(), &node->v());

    if (op->type() == op->v().type()) {
      *expr = op->v();
      return;
    }

#define __CAST_TO_TYPE(type__)                                          \
  if (auto* i = op->v().As<ir::IntImm>()) {                             \
    *expr = Expr(static_cast<type__>(i->value));                        \
  } else if (auto* f = op->v().As<ir::FloatImm>()) {                    \
    *expr = Expr(static_cast<type__>(NormCastValue<type__>(f->value))); \
  } else if (auto* u = op->v().As<ir::UIntImm>()) {                     \
    *expr = Expr(static_cast<type__>(u->value));                        \
  } else {                                                              \
    CINN_NOT_IMPLEMENTED                                                \
  }

    if (op->v().is_constant()) {
      if (op->type() == type_of<int8_t>()) {
        __CAST_TO_TYPE(int8_t)
      } else if (op->type() == type_of<int16_t>()) {
        __CAST_TO_TYPE(int16_t)
      } else if (op->type() == type_of<int32_t>()) {
        __CAST_TO_TYPE(int32_t)
      } else if (op->type() == type_of<int64_t>()) {
        __CAST_TO_TYPE(int64_t)
      } else if (op->type() == type_of<uint8_t>()) {
        __CAST_TO_TYPE(uint8_t)
      } else if (op->type() == type_of<uint16_t>()) {
        __CAST_TO_TYPE(uint16_t)
      } else if (op->type() == type_of<uint32_t>()) {
        __CAST_TO_TYPE(uint32_t)
      } else if (op->type() == type_of<uint64_t>()) {
        __CAST_TO_TYPE(uint64_t)
      } else if (op->type() == type_of<float>()) {
        __CAST_TO_TYPE(float)
      } else if (op->type() == type_of<double>()) {
        __CAST_TO_TYPE(double)
      } else if (op->type() == type_of<bool>()) {
        __CAST_TO_TYPE(bool)
      } else if (op->type() == type_of<uint32_t>()) {
        __CAST_TO_TYPE(uint32_t)
      } else if (op->type() == type_of<uint64_t>()) {
        __CAST_TO_TYPE(uint64_t)
      } else if (op->type() == type_of<bfloat16>()) {
        // Cannot simplify!!! pass
        __CAST_TO_TYPE(bfloat16)
      } else if (op->type() == type_of<float16>()) {
        // Cannot simplify!!! pass
        __CAST_TO_TYPE(float16)
      } else {
        CINN_NOT_IMPLEMENTED
      }
    }
#undef __CAST_TO_TYPE
  }
};
}  // namespace

class SimplifyNoPureMathPass : public ExprPass {
 public:
  SimplifyNoPureMathPass() : ExprPass("simplify_no_pure_math_pass") {}

  LogicalResult Run(ir::Expr* expr) override {
    SimplifyNoPureMathMutator mutator;
    mutator(expr);
    return LogicalResult::success();
  }
};
std::unique_ptr<ExprPass> CreateSimplifyNoPureMathPass() {
  return std::make_unique<SimplifyNoPureMathPass>();
}

class SimplifyLoadPass : public ExprPass {
 public:
  SimplifyLoadPass() : ExprPass("simplify_load_pass") {}

  LogicalResult Run(ir::Expr* expr) override {
    SimplifyLoadMutator mutator;
    mutator(expr);
    return LogicalResult::success();
  }
};
std::unique_ptr<ExprPass> CreateSimplifyLoadPass() {
  return std::make_unique<SimplifyLoadPass>();
}

class SimplifyStorePass : public BlockPass {
 public:
  SimplifyStorePass() : BlockPass("simplify_store_pass") {}

  LogicalResult Run(ir::stmt::BlockRef block) override {
    SimplifyStoreMutator mutator;
    mutator(block);
    return LogicalResult::success();
  }
};
std::unique_ptr<BlockPass> CreateSimplifyStorePass() {
  return std::make_unique<SimplifyStorePass>();
}

class SimplifySelectPass : public ExprPass {
 public:
  SimplifySelectPass() : ExprPass("simplify_select_pass") {}

  LogicalResult Run(ir::Expr* expr) override {
    SimplifySelectMutator mutator;
    mutator(expr);
    return LogicalResult::success();
  }
};
std::unique_ptr<ExprPass> CreateSimplifySelectPass() {
  return std::make_unique<SimplifySelectPass>();
}

class SimplifyForLoopsPass : public BlockPass {
 public:
  SimplifyForLoopsPass() : BlockPass("simplify_for_loops_pass") {}

  LogicalResult Run(ir::stmt::BlockRef block) override {
    SimplifyForLoopsMutator()(block);
    return LogicalResult::success();
  }
};

std::unique_ptr<BlockPass> CreateSimplifyForLoopsPass() {
  return std::make_unique<SimplifyForLoopsPass>();
}

class SimplifyIfThenElsePass : public BlockPass {
 public:
  SimplifyIfThenElsePass() : BlockPass("simplify_ifthenelse_loops_pass") {}

  LogicalResult Run(ir::stmt::BlockRef block) override {
    SimplifyIfThenElseMutator()(block);
    return LogicalResult::success();
  }
};

std::unique_ptr<BlockPass> CreateSimplifyIfThenElsePass() {
  return std::make_unique<SimplifyIfThenElsePass>();
}

class SimplifyRampPass : public ExprPass {
 public:
  SimplifyRampPass() : ExprPass("simplify_ramp_pass") {}

  LogicalResult Run(ir::Expr* expr) override {
    SimplifyRampMutator mutator;
    mutator(expr);
    return LogicalResult::success();
  }
};
std::unique_ptr<ExprPass> CreateSimplifyRampPass() {
  return std::make_unique<SimplifyRampPass>();
}

class SimplifyCastPass : public ExprPass {
 public:
  SimplifyCastPass() : ExprPass("simplify_cast_pass") {}

  LogicalResult Run(ir::Expr* expr) override {
    SimplifyCastMutator mutator;
    mutator(expr);
    return LogicalResult::success();
  }
};

std::unique_ptr<ExprPass> CreateSimplifyCastPass() {
  return std::make_unique<SimplifyCastPass>();
}

Expr ArithSimplify(const Expr& u) {
  if (!u.is_index()) return u;
  auto copied = ir_utils::IRCopy(u);
  return copied.as_index().Normalize();
}

void Simplify(ir::Expr* expr, const SimplifyType type) {
  VLOG(6) << "Begin Simplify: \n " << *expr;

  switch (type) {
    case SimplifyType::kExpr:
      SimplifyNoPureMathMutator()(expr);
      SimplifyCastMutator()(expr);
      SimplifyRampMutator()(expr);
      SimplifyLoadMutator()(expr);
      SimplifyLogicalMutator()(expr);
      SimplifySelectMutator()(expr);
      SimplifyNoPureMathMutator()(expr);
      break;
    case SimplifyType::kBlock:
      PADDLE_ENFORCE_EQ(expr->node_type(),
                        ir::Block::_node_type_,
                        ::common::errors::InvalidArgument(
                            "The Expr to simplify must be a block."));
      ir::stmt::BlockRef _block;

      try {
        // In the old IR, the body of a ScheduleBlock Expr may be a constant
        // (i.e. an integer, '0'), which can not be converted to a statment.
        _block = ir::ConvertExprBlockToStmtBlock(*expr);
      } catch (const ::common::enforce::EnforceNotMet& e) {
        LOG(WARNING) << "ConvertExprBlockToStmtBlock failed! Error:"
                     << e.what();
        return;
      }
      optim::ExprPassManager expr_pass_manager;
      expr_pass_manager.AddPass(CreateSimplifyNoPureMathPass());
      expr_pass_manager.AddPass(CreateSimplifyCastPass());
      expr_pass_manager.AddPass(CreateSimplifyRampPass());
      expr_pass_manager.AddPass(CreateSimplifyLoadPass());
      expr_pass_manager.AddPass(CreateSimplifyLogicalPass());
      expr_pass_manager.AddPass(CreateSimplifySelectPass());
      expr_pass_manager.Run(_block);

      optim::BlockPassManager block_pass_manager;
      block_pass_manager.AddPass(CreateSimplifyStorePass());
      block_pass_manager.AddPass(CreateSimplifyIfThenElsePass());
      block_pass_manager.Run(_block);

      optim::ExprPassManager expr_pass_manager1;
      expr_pass_manager1.AddPass(CreateSimplifyNoPureMathPass());
      expr_pass_manager1.AddPass(CreateSimplifyFracWithDivPass());
      expr_pass_manager1.Run(_block);

      *expr = ir::ConvertStmtBlockToExprBlock(_block);
      break;
  }

  VLOG(6) << "End Simplify: \n" << *expr;
}

void SimplifyForLoops(ir::Expr* expr) {
  PADDLE_ENFORCE_EQ(expr->node_type(),
                    ir::Block::_node_type_,
                    ::common::errors::InvalidArgument(
                        "The Expr to simplify must be a block."));
  ir::stmt::BlockRef block = ir::ConvertExprBlockToStmtBlock(*expr);
  optim::BlockPassManager pass_manager;
  pass_manager.AddPass(CreateSimplifyForLoopsPass());
  pass_manager.Run(block);
  *expr = ir::ConvertStmtBlockToExprBlock(block);
}

void SimplifyCast(ir::Expr* expr) {
  SimplifyCastMutator mutator;
  mutator(expr);
}

void SimplifyCast(ir::stmt::BlockRef block) {
  optim::ExprPassManager pass_manager;
  pass_manager.AddPass(CreateSimplifyCastPass());
  pass_manager.Run(block);
}

}  // namespace optim
}  // namespace cinn
