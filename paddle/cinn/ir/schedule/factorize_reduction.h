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

// Used in FactorizeReduction

#pragma once
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/utils/error.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {

// Create the new Reduction-Factorized tensor,
// only used for FactorizeReduction schedule primitive.
Tensor CreateRFTensor(const Tensor& original_tensor,
                      const Expr& rf_loop,
                      int rf_axis) {
  std::string name = common::UniqName(original_tensor->name + "_rf");
  std::vector<Expr> new_shape = original_tensor->shape;
  new_shape.insert(new_shape.begin() + rf_axis, rf_loop.As<For>()->extent);
  Tensor rf_tensor = _Tensor_::Make(name,
                                    original_tensor->type(),
                                    new_shape,
                                    new_shape,
                                    original_tensor->operation,
                                    original_tensor->reduce_axis);
  rf_tensor->WithBuffer("global", name, original_tensor->type());
  return rf_tensor;
}

// Base class to create a new reduce block,
// only used for FactorizeReduction schedule primitive.
class ReduceBlockCreator {
 public:
  ReduceBlockCreator(const Expr& original_block,
                     const std::vector<Expr>& original_loops,
                     const Expr& rf_loop,
                     const Expr& original_update_stmt,
                     const ir::Tensor& rf_tensor,
                     bool is_rf_block)
      : original_block_(original_block),
        original_loops_(original_loops),
        rf_loop_(rf_loop),
        original_update_stmt_(original_update_stmt),
        rf_tensor_(rf_tensor),
        is_rf_block_(is_rf_block) {
    const ScheduleBlockRealize* block_real =
        original_block_.As<ir::ScheduleBlockRealize>();
    PADDLE_ENFORCE_NOT_NULL(block_real,
                            ::common::errors::InvalidArgument(
                                "The block is not a ScheduleBlockRealize"));
    num_block_iters_ = block_real->iter_values.size();
  }

  void CreateBlock() {
    CreateRFIter();
    for (int i = 0; i < num_block_iters_; ++i) {
      CreateNormalIter(i);
    }
    CreateUpdateStmt();

    std::string new_update_block_name =
        original_block_.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>()
            ->name;
    if (is_rf_block_) {
      new_update_block_name = rf_tensor_->name;
    }
    std::string new_init_block_name =
        ir::GenReduceInitTensorNameOf(new_update_block_name);
    VLOG(5) << "new_init_block_name = " << new_init_block_name;

    const ir::Tensor& real_tensor =
        is_rf_block_
            ? rf_tensor_
            : original_update_stmt_.As<ir::Store>()->tensor.as_tensor_ref();

    Expr init_value = real_tensor->GetReduceInitVal();
    const std::vector<Expr>& domain = real_tensor->domain_without_reduce_axis();
    ir::Tensor init_tensor = lang::Compute(
        domain,
        [=](const std::vector<Expr>& axis) { return init_value; },
        new_init_block_name);
    init_tensor->Bind(real_tensor->buffer);
    std::vector<Expr> new_indices;
    if (new_update_stmt_.As<ir::Store>()) {
      new_indices = new_update_stmt_.As<ir::Store>()->indices;
    } else if (new_update_stmt_.As<ir::IfThenElse>()) {
      new_indices = new_update_stmt_.As<ir::IfThenElse>()
                        ->true_case.As<ir::Block>()
                        ->stmts[0]
                        .As<ir::Store>()
                        ->indices;
    } else {
      throw std::runtime_error("only support store and ifthenelse");
    }

    Expr init_stmt = ir::Store::Make(init_tensor, init_value, new_indices);

    new_init_sch_block_ = ScheduleBlock::Make(
        new_init_iter_vars_, {}, {}, new_init_block_name, init_stmt);
    new_init_block_realize_ =
        ScheduleBlockRealize::Make(new_init_iter_values_, new_init_sch_block_);

    new_update_sch_block_ = ScheduleBlock::Make(
        new_iter_vars_, {}, {}, new_update_block_name, new_update_stmt_);
    new_update_block_realize_ =
        ScheduleBlockRealize::Make(new_iter_values_, new_update_sch_block_);
    VLOG(4) << "new_update_block_realize:\n" << new_update_block_realize_;
  }

  Expr CreateLoops(bool with_init = true) {
    int num_loops = original_loops_.size();
    std::vector<Expr> new_loops(num_loops);
    Expr body = new_update_block_realize_;
    bool has_add_init_block = false;
    // `is_inside_rf_loop` is used to skip loop inside rf_loop.
    bool is_inside_rf_loop = true;
    for (int i = num_loops - 1; i >= 0; --i) {
      bool is_spatial_loop =
          new_spatial_loop_var_names_.count(
              original_loops_[i].As<For>()->loop_var->name) > 0;
      bool is_rf_loop = rf_loop_.As<For>()->loop_var->name ==
                        original_loops_[i].As<For>()->loop_var->name;
      // Outer loop should not skip.
      if (is_rf_loop) {
        is_inside_rf_loop = false;
      }
      // Skip non rf reduction loops of write back block.
      if (!is_rf_block_ && is_inside_rf_loop && !is_spatial_loop) {
        continue;
      }
      // Add reduce init block.
      if (!has_add_init_block && is_spatial_loop && with_init) {
        body = Block::Make({new_init_block_realize_, body});
        has_add_init_block = true;
      }
      // Add If
      if (original_loops_[i].As<For>()->body.As<IfThenElse>()) {
        const IfThenElse* original_if =
            original_loops_[i].As<For>()->body.As<IfThenElse>();
        body = IfThenElse::Make(original_if->condition, body);
      }
      if (original_loops_[i].As<For>()->body.As<Block>() &&
          original_loops_[i].As<For>()->body.As<Block>()->stmts.size() == 1 &&
          original_loops_[i]
              .As<For>()
              ->body.As<Block>()
              ->stmts[0]
              .As<IfThenElse>()) {
        const IfThenElse* original_if = original_loops_[i]
                                            .As<For>()
                                            ->body.As<Block>()
                                            ->stmts[0]
                                            .As<IfThenElse>();
        body = IfThenElse::Make(original_if->condition, body);
      }
      // Add loops
      Var loop_var = ir_utils::IRCopy(original_loops_[i].As<For>()->loop_var);
      Expr min = ir_utils::IRCopy(original_loops_[i].As<For>()->min);
      Expr extent = ir_utils::IRCopy(original_loops_[i].As<For>()->extent);
      body = For::Make(loop_var,
                       min,
                       extent,
                       original_loops_[i].As<For>()->for_type(),
                       original_loops_[i].As<For>()->device_api,
                       body,
                       original_loops_[i].As<For>()->vectorize_info(),
                       original_loops_[i].As<For>()->bind_info());
      VLOG(5) << "new body:\n" << body;
    }
    VLOG(4) << "new loop nest:\n" << body;
    return body;
  }

 private:
  virtual void CreateRFIter() = 0;
  virtual void CreateNormalIter(int idx) = 0;
  virtual void CreateUpdateStmt() = 0;

 public:
  Var rf_var_;
  std::vector<Expr> rf_tensor_access_indices_;

 protected:
  const Expr& original_block_;
  const std::vector<Expr>& original_loops_;
  const Expr& rf_loop_;
  const Expr& original_update_stmt_;
  const ir::Tensor& rf_tensor_;
  std::map<Var, Expr, CompVar> original_indice2new_expr_;
  int num_block_iters_;
  bool is_rf_block_;

  std::vector<Var> new_iter_vars_;
  std::vector<Expr> new_iter_values_;
  std::vector<Var> new_init_iter_vars_;
  std::vector<Expr> new_init_iter_values_;
  std::unordered_set<std::string> new_spatial_loop_var_names_;
  Expr new_update_stmt_;

  Expr new_update_sch_block_;
  Expr new_update_block_realize_;
  Expr new_init_sch_block_;
  Expr new_init_block_realize_;
};

class LoadReplacer : public ir::IRMutator<> {
 public:
  explicit LoadReplacer(const std::string& src_load_tensor_name,
                        const ir::Expr& target)
      : src_load_tensor_name_(src_load_tensor_name), target_(target) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::Load* expr, Expr* op) override {
    if (expr->tensor.as_tensor()->name == src_load_tensor_name_) {
      *op = target_;
    }
  }

 private:
  std::string src_load_tensor_name_;
  ir::Expr target_;
};

// Implement class for building Reduction-Factorized block,
// only used for FactorizeReduction schedule primitive.
class RFBlockCreator : public ReduceBlockCreator {
 public:
  RFBlockCreator(const Expr& original_block,
                 const std::vector<Expr>& original_loops,
                 const Expr& rf_loop,
                 const Expr& original_update_stmt,
                 const ir::Tensor& rf_tensor,
                 const std::map<Var, Expr, CompVar>& var2loops,
                 const Expr& bound_check,
                 int rf_axis)
      : ReduceBlockCreator(original_block,
                           original_loops,
                           rf_loop,
                           original_update_stmt,
                           rf_tensor,
                           true),
        var2loops_(var2loops),
        rf_axis_(rf_axis),
        bound_check_(ir_utils::IRCopy(bound_check)) {}

 private:
  void CreateRFIter() override {
    std::string loop_var_name = rf_loop_.As<ir::For>()->loop_var->name;
    std::string rf_var_name = common::UniqName("v" + loop_var_name);
    rf_var_ = Var(rf_loop_.As<ir::For>()->min,
                  rf_loop_.As<ir::For>()->extent,
                  rf_var_name,
                  /* is_reduce = */ false);
    loop_var2block_iters_[rf_loop_.As<ir::For>()->loop_var] = rf_var_;
    new_iter_vars_.push_back(rf_var_);
    new_iter_values_.push_back(rf_loop_.As<ir::For>()->loop_var);
    new_init_iter_vars_.push_back(rf_var_);
    new_init_iter_values_.push_back(rf_loop_.As<ir::For>()->loop_var);
    new_spatial_loop_var_names_.insert(rf_loop_.As<ir::For>()->loop_var->name);

    std::vector<Expr> new_iter_exprs{Expr(rf_var_)};
    ReplaceExpr(
        &bound_check_, {rf_loop_.As<ir::For>()->loop_var}, new_iter_exprs);

    VLOG(4) << "create new_rf_var = " << rf_var_
            << ", with iter value = " << new_iter_values_.back();
  }

  void CreateNormalIter(int idx) override {
    Var original_iter_var = original_block_.As<ir::ScheduleBlockRealize>()
                                ->schedule_block.As<ir::ScheduleBlock>()
                                ->iter_vars[idx];
    Expr original_iter_value =
        original_block_.As<ir::ScheduleBlockRealize>()->iter_values[idx];
    // The original iter is either a spatial iter, or a reduction iter that
    // doesn't touch the rf loop. In this case reuse the old iter var and its
    // corresponding iter value.
    if (!original_iter_var->is_reduce_axis) {
      new_iter_vars_.push_back(original_iter_var);
      new_iter_values_.push_back(original_iter_value);
      new_init_iter_vars_.push_back(original_iter_var);
      new_init_iter_values_.push_back(original_iter_value);
      ir_utils::CollectIRNodesWithoutTensor(
          original_iter_value, [&](const Expr* x) {
            if (x->as_var()) {
              new_spatial_loop_var_names_.insert(x->as_var()->name);
            }
            return false;
          });
      return;
    } else if (!ContainVar({original_iter_value},
                           rf_loop_.As<ir::For>()->loop_var->name)) {
      new_iter_vars_.push_back(original_iter_var);
      new_iter_values_.push_back(original_iter_value);
      return;
    }
    PADDLE_ENFORCE_EQ(
        original_iter_var->is_reduce_axis,
        true,
        ::common::errors::InvalidArgument(
            "The original_iter_var is expected to be a reduce axis."));

    // This iter is a reduction iter and touches the rfactor loop. So we try to
    // create a new iter for each loop var that appear in the original iter
    // value.
    std::vector<Var> vars_in_original_iter_values;
    ir_utils::CollectIRNodesWithoutTensor(
        original_iter_value, [&](const Expr* x) {
          if (x->as_var()) {
            vars_in_original_iter_values.push_back(x->as_var_ref());
          }
          return false;
        });
    for (const Var& loop_var : vars_in_original_iter_values) {
      if (var2loops_.count(loop_var) == 0) {
        continue;
      }
      Expr loop = var2loops_.at(loop_var);
      if (loop_var2block_iters_.count(loop_var) == 0) {
        Var new_iter_var(loop.As<ir::For>()->min,
                         loop.As<ir::For>()->extent,
                         common::UniqName("v" + loop_var->name),
                         /* is_reduce = */ true);
        new_iter_vars_.push_back(new_iter_var);
        new_iter_values_.emplace_back(loop_var);
        loop_var2block_iters_[loop_var] = new_iter_var;
      }
    }
    // Substitute the original iter values with new iter vars,
    // and store the new iter values in original_indice2new_expr_,
    // it will be used in Load/Store indices.
    Expr new_iters = ir_utils::IRCopy(original_iter_value);
    ReplaceExpr(&new_iters, loop_var2block_iters_);
    original_indice2new_expr_[original_iter_var] = new_iters;
    VLOG(4) << "original_indice2new_expr_[" << original_iter_var
            << "] = " << new_iters;
  }

  void CreateUpdateStmt() override {
    rf_tensor_access_indices_ = original_update_stmt_.As<ir::Store>()->indices;
    rf_tensor_access_indices_.insert(
        rf_tensor_access_indices_.begin() + rf_axis_, rf_var_);
    Expr original_store_body = original_update_stmt_.As<ir::Store>()->value;
    std::string original_store_name =
        original_update_stmt_.As<ir::Store>()->tensor.as_tensor()->name;
    Expr new_store_body = ir_utils::IRCopy(original_store_body);
    LoadReplacer load_replacer(
        original_store_name, Load::Make(rf_tensor_, rf_tensor_access_indices_));
    load_replacer(&new_store_body);

    new_update_stmt_ =
        ir::Store::Make(rf_tensor_, new_store_body, rf_tensor_access_indices_);

    if (!bound_check_.is_constant()) {
      new_update_stmt_ = ir::IfThenElse::Make(bound_check_, new_update_stmt_);
    }
    ReplaceExpr(&new_update_stmt_, original_indice2new_expr_);
    VLOG(4) << "new_update_stmt of rf block: \n" << new_update_stmt_;
  }

 private:
  const std::map<Var, Expr, CompVar>& var2loops_;
  int rf_axis_;

  std::map<Var, Expr, CompVar> loop_var2block_iters_;

  Expr bound_check_;
};

// Implement class for building Writing-Back block,
// only used for FactorizeReduction schedule primitive.
class RBBlockCreator : public ReduceBlockCreator {
 public:
  RBBlockCreator(const Expr& original_block,
                 const std::vector<Expr>& original_loops,
                 const Expr& rf_loop,
                 const Expr& original_update_stmt,
                 const ir::Tensor& rf_tensor,
                 const std::vector<Expr>& rf_tensor_access_indices,
                 const Var& rf_block_rf_iter_var)
      : ReduceBlockCreator(original_block,
                           original_loops,
                           rf_loop,
                           original_update_stmt,
                           rf_tensor,
                           false),
        rf_tensor_access_indices_(rf_tensor_access_indices),
        rf_block_rf_iter_var_(rf_block_rf_iter_var) {}

 private:
  void CreateRFIter() override {
    std::string loop_var_name = rf_loop_.As<ir::For>()->loop_var->name;
    std::string rf_var_name = common::UniqName("v" + loop_var_name);
    rf_var_ = Var(rf_loop_.As<ir::For>()->min,
                  rf_loop_.As<ir::For>()->extent,
                  rf_var_name,
                  /* is_reduce = */ true);
    new_iter_vars_.push_back(rf_var_);
    new_iter_values_.push_back(rf_loop_.As<ir::For>()->loop_var);
    original_indice2new_expr_[rf_block_rf_iter_var_] = Expr(rf_var_);
    VLOG(4) << "create new_rf_var = " << rf_var_
            << ", with iter value = " << new_iter_values_.back();
  }

  void CreateNormalIter(int idx) override {
    Var original_iter_var = original_block_.As<ir::ScheduleBlockRealize>()
                                ->schedule_block.As<ir::ScheduleBlock>()
                                ->iter_vars[idx];
    Expr original_iter_value =
        original_block_.As<ir::ScheduleBlockRealize>()->iter_values[idx];
    if (!original_iter_var->is_reduce_axis) {
      new_iter_vars_.push_back(original_iter_var);
      new_iter_values_.push_back(original_iter_value);
      new_init_iter_vars_.push_back(original_iter_var);
      new_init_iter_values_.push_back(original_iter_value);
      ir_utils::CollectIRNodesWithoutTensor(
          original_iter_value, [&](const Expr* x) {
            if (x->as_var()) {
              new_spatial_loop_var_names_.insert(x->as_var()->name);
            }
            return false;
          });
      // original_indice2new_expr_[original_iter_var] = new_iter_vars_.back();
      VLOG(4) << "create new iter var = " << new_iter_vars_.back()
              << ", with iter value = " << new_iter_values_.back();
    }
  }

  void CreateUpdateStmt() override {
    Expr original_store_body = original_update_stmt_.As<ir::Store>()->value;
    Expr new_store_body = ir_utils::IRCopy(original_store_body);
    std::string original_store_name =
        original_update_stmt_.As<ir::Store>()->tensor.as_tensor()->name;

#define REPLACE_RF_TENSOR(Op)                                          \
  if (new_store_body.As<Op>()) {                                       \
    auto* node = new_store_body.As<Op>();                              \
    PADDLE_ENFORCE_NOT_NULL(node,                                      \
                            ::common::errors::InvalidArgument(         \
                                "The conversion of new_store_body to " \
                                "Op* failed, node is nullptr."));      \
    auto& operand = node->b();                                         \
    operand = Load::Make(rf_tensor_, rf_tensor_access_indices_);       \
  }

    REPLACE_RF_TENSOR(Add)
    REPLACE_RF_TENSOR(Mul)
    REPLACE_RF_TENSOR(Max)
    REPLACE_RF_TENSOR(Min)
    REPLACE_RF_TENSOR(And)
    REPLACE_RF_TENSOR(Or)
    REPLACE_RF_TENSOR(LT)
    REPLACE_RF_TENSOR(LE)
    REPLACE_RF_TENSOR(GT)
    REPLACE_RF_TENSOR(GE)
#undef REPLACE_RF_TENSOR

    Expr original_store_tensor = original_update_stmt_.As<ir::Store>()->tensor;
    std::vector<Expr> original_store_indices =
        original_update_stmt_.As<ir::Store>()->indices;
    new_update_stmt_ = ir::Store::Make(
        original_store_tensor, new_store_body, original_store_indices);
    ReplaceExpr(&new_update_stmt_, original_indice2new_expr_);
    VLOG(4) << "new_update_stmt of write back block: \n" << new_update_stmt_;
  }

 private:
  const std::vector<Expr>& rf_tensor_access_indices_;
  const Var& rf_block_rf_iter_var_;
};

}  // namespace ir
}  // namespace cinn
