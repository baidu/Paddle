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
namespace optim {

enum class SimplifyType { kExpr, kBlock };

/**
 * Simplify the expression on Cast, Ramp, Load, Store, IfThenElse and Select
 * operations.
 *
 * This pass is applicable in scenarios where expressions contain redundant
 * operations or constants that can be simplified. This is common in
 * mathematical computations where certain patterns, such as adding zero or
 * multiplying by zero, occur frequently.

 * When applied, this pass will traverse the expression and simplify it by
 * applying mathematical identities, such as removing zero in addition or
 * multiplication, and combining terms in polynomial expressions.

 * Performance impact: This pass addresses the performance issues related to
 * unnecessary computations in expressions, which can lead to reduced execution
 * time and improved efficiency in code generation.

 * Examples:
 * 1. Basic simplification:
 *    Input IR:
 *      A + 0
 *    Output IR:
 *      A
 *
 * 2. Polynomial simplification:
 *    Input IR:
 *      A[i * 0 + 2 * a + 3 * a + 1 + 2]
 *    Output IR:
 *      A[5 * a + 3]
 *
 * 3. Ramp simplification:
 *    Input IR:
 *      Add([1, 3, 5, 7], [2, 4, 6, 8])
 *    Output IR:
 *      [3, 7, 11, 15]
 *
 * 4. Load simplification:
 *    Input IR:
 *      Load(buffer, {i + 0, j * 1})
 *    Output IR:
 *      Load(buffer, {i, j})
 *
 * 5. Store simplification:
 *    Input IR:
 *      Store(buffer, value, {i + 0, j * 1})
 *    Output IR:
 *      Store(buffer, value, {i, j})
 *
 * 6. IfThenElse simplification:
 *    Input IR:
 *      If(1+2):
 *        If(false):
 *          ... (true_case_1)
 *        Else:
 *          ... (false_case_1)
 *      Else:
 *        ... (false_case_2)
 *    Output IR:
 *      ... (false_case_1)
 *
 * 7. Select simplification:
 *    Input IR:
 *      Select(1+2, true_value, false_value)
 *    Output IR:
 *      true_value
 */
void Simplify(ir::Expr *expr, SimplifyType type);

/**
 * Simplify type casting expressions.
 *
 * This pass is applicable when type casting operations in the IR is possible to
 * be simplified or eliminated. It is particularly useful in scenarios where
 * the type of the expression being cast is already known and can be directly
 * used.

 * When applied, this pass will check if the cast type matches the type of the
 * value to be be cast. If they are the same, the cast will be removed to
 * simplify the cast expression. Additionally, if the value being cast is a
 * constant, the cast expression will be simplified to the cast type as well.

 * Performance impact: This pass addresses performance issues related to
 * unnecessary type casting, which can lead to improved runtime efficiency by
 * reducing overhead.

 * Examples:
 * 1. Redundant cast removal:
 *    Input IR:
 *      Cast<int>(5)
 *    Output IR:
 *      5
 *
 * 2. Type mismatch:
 *    Input IR:
 *      int x = 5
 *      Cast<float>(x)
 *      Cast<float>(5)
 *    Output IR:
 *      Cast<float>(x)  (Type mismatch, remains unchanged)
 *      5.0             (Constant value will be cast)
 */
void SimplifyCast(ir::Expr *expr);

/**
 * Simplify for loop structures in the IR.
 *
 * This pass is applicable in scenarios where for loops are trivial, such as
 * loops that iterate exactly once. This simplification is important for
 * optimizing loops in high-performance computing scenarios.

 * When applied, this pass will check for for loops that have a constant extent
 * of 1 and will replace them with their body, effectively removing the loop
 * and simplifying the IR.

 * Performance impact: This pass can lead to significant performance
 * improvements by eliminating unnecessary loop overhead and allowing
 * for better optimization of the loop body.

 * Examples:
 * 1. Trivial loop simplification:
 *    Input IR:
 *      for (int i = 1; i < 2; ++i) { doSomething(i); }
 *    Output IR:
 *      doSomething(1)
 *
 * 2. Non-trivial loop:
 *    Input IR:
 *      for (int i = 0; i < 2; ++i) { doSomething(i); }
 *    Output IR:
 *      for (int i = 0; i < 2; ++i) { doSomething(i); } (remains unchanged)
 */
void SimplifyForLoops(ir::Expr *expr);

Expr ArithSimplify(const Expr &u);
}  // namespace optim
}  // namespace cinn
