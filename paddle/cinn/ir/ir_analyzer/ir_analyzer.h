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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"

namespace cinn {
namespace ir {
namespace analyzer {

bool HasBlock(const std::vector<Expr>& exprs, const std::string& block_name);

std::vector<Expr> GetLoops(const std::vector<Expr>& exprs,
                           const std::string& block_name);

std::vector<Expr> GetLoops(const std::vector<Expr>& exprs, const Expr& block);

std::vector<Expr> GetAllBlocks(const std::vector<Expr>& exprs);

std::vector<Expr> GetChildBlocks(const Expr& expr);

Expr GetBlock(const std::vector<Expr>& exprs, const std::string& block_name);

/**
 * Get the root schedule block (i.e. ScheduleBlock(root)) from `expr`.
 * The `expr` must be the root block of ModuleExpr.
 */
Expr GetRootSBlock(const Expr& expr);

Expr GetRootBlock(const std::vector<Expr>& exprs, const Expr& expr);

DeviceAPI GetDeviceAPI(const std::vector<Expr>& exprs);

Expr AddUnitLoop(const std::vector<Expr>& exprs, const Expr& block);

Expr GetStoreOfSBlock(const Expr& block);

Tensor GetStoreTensorOfSBlock(const Expr& block);

std::vector<Expr> GetConsumerSBlocks(const Expr& block, const Expr& root);

std::vector<std::pair<Expr, Expr>> GetConsumerLoadsAndSBlocks(const Expr& block,
                                                              const Expr& root);

std::unordered_map<std::string, std::unordered_map<ir::Var, ir::Expr>>
CollectVarToForMap(const std::vector<Expr>& exprs,
                   const std::vector<Expr>& blocks);

std::unordered_map<ir::Var, ir::Expr> GetIterVarToValueOfSBlock(ir::Expr block);

ir::Expr ReplaceVarWithExpr(const ir::Expr& source,
                            const std::vector<ir::Var>& candidates,
                            const std::vector<ir::Expr>& targets);

/**
 * Expand the iter_vars in `expr` to the iter_values of `block`.
 */
Expr ExpandIterVar(const Expr& expr, const Expr& block);

constexpr char* kLoopVar = "loop_var_";

/**
 * Replace the loop_vars in `expr` to the canonicalized form such that the
 * loop_var of loop[i] has name `loop_var_i`.
 */
Expr CanonicalizeLoopVar(const Expr& expr, const std::vector<Expr>& loops);

std::vector<ir::Expr> GetIterValuesOfAccess(ir::Expr load_or_store,
                                            ir::Expr block);

std::unordered_set<ir::Var> GetReduceIterVars(ir::Expr block);

bool IsReductionSBlock(ir::Expr block);

bool IsBroadcastSBlock(ir::Expr block);

std::vector<ir::Var> IndicesToVars(const std::vector<ir::Expr>& indices);

void AnalyzeScheduleBlockReadWriteBuffer(ir::ScheduleBlock* sche_block);

std::string GetBlockName(const ir::Expr block);

bool IsVectorizable(const std::vector<ir::Expr>& loops,
                    int vectorize_factor,
                    int current_reduce_axis,
                    int sp_thread,
                    int rd_thread,
                    bool is_reduce_axis);

}  // namespace analyzer
}  // namespace ir
}  // namespace cinn
