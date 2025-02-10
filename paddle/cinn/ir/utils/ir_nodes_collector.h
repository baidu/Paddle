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

#pragma once

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace ir {
namespace ir_utils {
/**
 * Collect the IR Nodes(without duplication) in the expression.
 */
std::set<Expr> CollectIRNodes(Expr x,
                              std::function<bool(const Expr*)>&& teller,
                              bool uniq_target = false);

std::set<Expr> CollectIRNodes(ir::LoweredFunc f,
                              std::function<bool(const Expr*)>&& teller,
                              bool uniq_target = false);

std::set<Expr> CollectIRNodes(ir::stmt::BlockRef block,
                              std::function<bool(const Expr*)>&& teller,
                              bool uniq_target = false);

/**
 * Collect the IR Nodes(without duplication and tensor's compute body) in the
 * expression.
 */
std::vector<Expr> CollectIRNodesWithoutTensor(
    Expr x,
    std::function<bool(const Expr*)>&& teller,
    bool uniq_target = false);

/**
 * Collect the IR Nodes from Block.
 */
std::vector<Expr> CollectIRNodesInOrder(
    Expr block, std::function<bool(const Expr*)>&& teller);

/**
 * Collect the tensors in Load nodes.
 */
std::set<Expr> CollectLoadTensors(Expr x,
                                  std::function<bool(const Expr*)>&& teller);

/**
 * Collect the tensors in Store nodes.
 */
std::set<Expr> CollectStoreTensors(Expr x,
                                   std::function<bool(const Expr*)>&& teller);

/**
 * Collect both the Store and Load nodes.
 */
std::set<Expr> CollectReferencedTensors(
    Expr x, const std::function<bool(const Expr*)>& teller);

std::map<std::string, Expr> CollectTensorMap(
    Expr x,
    std::function<bool(const Expr*)>&& extra_teller = [](const Expr* x) {
      return true;
    });

/**
 * Collect undefined vars in the scope.
 *
 * e.g.
 *
 * The expression:
 * for i
 *  for j
 *    a[i, j] = b[i, j]
 *
 * here a, b are vars without definition
 */
std::vector<std::string> CollectUndefinedVars(const Expr* e);

/**
 * Collect the Tensor Nodes which will be written by Store or Call Nodes
 */
std::set<std::string> CollectTensorNeedsWrite(const Expr* e);

/**
 * Collect the Tensor Nodes which will be written by Store or Call Nodes
 */
std::set<std::string> CollectTensorNeedsWrite(const stmt::BlockRef& block);
}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
