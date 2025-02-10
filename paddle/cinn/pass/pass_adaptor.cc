// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include "paddle/cinn/ir/expr_visitors.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/pass/pass_adaptor.h"

namespace cinn {
namespace optim {
namespace detail {
namespace {
template <typename PassT, typename IRScopeRefT>
LogicalResult RunPasses(const std::vector<std::unique_ptr<PassT>>& passes,
                        IRScopeRefT scope) {
  for (auto& pass : passes) {
    if ((pass->Run(scope)).failed()) {
      VLOG(3) << "Failed to run pass: " << pass->name();
      return LogicalResult::failure();
    }
  }
  return LogicalResult::success();
}
}  // namespace

LogicalResult FuncPassAdaptor::Run(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<FuncPass>>& passes) {
  return RunPasses(passes, func);
}

LogicalResult FuncPassAdaptor::Run(
    ir::stmt::BlockRef block,
    const std::vector<std::unique_ptr<FuncPass>>& passes) {
  return RunPasses(passes, block);
}

LogicalResult BlockPassAdaptor::Run(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<BlockPass>>& passes) {
  return Run(func->body_block, passes);
}

LogicalResult BlockPassAdaptor::Run(
    ir::stmt::BlockRef block,
    const std::vector<std::unique_ptr<BlockPass>>& passes) {
  std::vector<ir::stmt::StmtRef> new_stmts = block->stmts();
  for (ir::stmt::StmtRef inner_stmt : new_stmts) {
    std::vector<ir::stmt::BlockRef> inner_blocks = inner_stmt->block_fields();
    for (ir::stmt::BlockRef inner_block : inner_blocks) {
      if (Run(inner_block, passes).failed()) return LogicalResult::failure();
    }
    inner_stmt->set_block_fields(inner_blocks);
  }
  block->set_stmts(new_stmts);
  return RunPasses(passes, block);
}

LogicalResult StmtPassAdaptor::Run(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<StmtPass>>& passes) {
  return Run(func->body_block, passes);
}

LogicalResult StmtPassAdaptor::Run(
    ir::stmt::BlockRef block,
    const std::vector<std::unique_ptr<StmtPass>>& passes) {
  LogicalResult res = LogicalResult::success();
  ir::stmt::Mutate(
      block,
      [&](ir::stmt::StmtRef stmt) {},
      [&](ir::stmt::StmtRef stmt) {
        if (RunPasses(passes, stmt).failed()) {
          res = LogicalResult::failure();
        }
      });
  return res;
}

namespace {
using ExprMutateFuncT = std::function<LogicalResult(ir::Expr* expr)>;
class StmtToExprPassAdaptor : public StmtPass {
 public:
  explicit StmtToExprPassAdaptor(const ExprMutateFuncT& func)
      : StmtPass("stmt to expr pass adaptor"), mutator_(func) {}
  virtual LogicalResult Run(ir::stmt::StmtRef stmt) {
    return mutator_.VisitStmt(stmt);
  }

 private:
  class LocalExprMutator : public ir::stmt::StmtMutator<LogicalResult> {
   public:
    explicit LocalExprMutator(const ExprMutateFuncT& expr_mutator)
        : expr_mutator_(expr_mutator) {}

    LogicalResult VisitStmt(ir::stmt::StmtRef stmt) override {
      return ir::stmt::StmtMutator<LogicalResult>::VisitStmt(stmt);
    }

   private:
    ExprMutateFuncT expr_mutator_;
#define __(stmt__) LogicalResult VisitStmt(ir::stmt::stmt__ stmt) override;
    NODETY_FORALL_STMT(__)
#undef __
  };
  LocalExprMutator mutator_;
};

#define __(stmt__)                                                  \
  LogicalResult StmtToExprPassAdaptor::LocalExprMutator::VisitStmt( \
      ir::stmt::stmt__ stmt) {                                      \
    LogicalResult res = LogicalResult::success();                   \
    const auto& MutateFunc = [&](ir::Expr* expr) {                  \
      if (expr_mutator_(expr).failed()) {                           \
        res = LogicalResult::failure();                             \
      }                                                             \
    };                                                              \
    ir::MutateExpr(stmt, MutateFunc);                               \
    return res;                                                     \
  }
NODETY_FORALL_STMT(__)
#undef __
}  // namespace

LogicalResult ExprPassAdaptor::Run(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<ExprPass>>& passes) {
  return Run(func->body_block, passes);
}

LogicalResult ExprPassAdaptor::Run(
    ir::stmt::BlockRef block,
    const std::vector<std::unique_ptr<ExprPass>>& passes) {
  std::vector<std::unique_ptr<StmtPass>> stmt_passes;
  stmt_passes.emplace_back(std::move(std::make_unique<StmtToExprPassAdaptor>(
      [&](ir::Expr* expr) { return RunPasses(passes, expr); })));
  StmtPassAdaptor stmt_pass_adaptor;
  return stmt_pass_adaptor.Run(block, stmt_passes);
}

}  // namespace detail
}  // namespace optim
}  // namespace cinn
