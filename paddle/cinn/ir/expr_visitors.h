// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace ir {

// Defines utilities for walking exprs (no walking into nest block)
void VisitExpr(const stmt::Let &stmt,
               const std::function<void(const Expr &)> &callback);

void VisitExpr(const stmt::Store &stmt,
               const std::function<void(const Expr &)> &callback);

void VisitExpr(const stmt::Alloc &stmt,
               const std::function<void(const Expr &)> &callback);

void VisitExpr(const stmt::Free &stmt,
               const std::function<void(const Expr &)> &callback);

void VisitExpr(const stmt::IfThenElse &stmt,
               const std::function<void(const Expr &)> &callback);

void VisitExpr(const stmt::For &stmt,
               const std::function<void(const Expr &)> &callback);

void VisitExpr(const stmt::Schedule &stmt,
               const std::function<void(const Expr &)> &callback);

void VisitExpr(const stmt::Evaluate &stmt,
               const std::function<void(const Expr &)> &callback);

void MutateExpr(stmt::Let stmt, const std::function<void(Expr *)> &callback);

void MutateExpr(stmt::Store stmt, const std::function<void(Expr *)> &callback);

void MutateExpr(stmt::Alloc stmt, const std::function<void(Expr *)> &callback);

void MutateExpr(stmt::Free stmt, const std::function<void(Expr *)> &callback);

void MutateExpr(stmt::IfThenElse stmt,
                const std::function<void(Expr *)> &callback);

void MutateExpr(stmt::For stmt, const std::function<void(Expr *)> &callback);

void MutateExpr(stmt::Schedule stmt,
                const std::function<void(Expr *)> &callback);

void MutateExpr(stmt::Evaluate stmt,
                const std::function<void(Expr *)> &callback);

}  // namespace ir
}  // namespace cinn
