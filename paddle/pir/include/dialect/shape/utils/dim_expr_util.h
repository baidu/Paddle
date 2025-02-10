// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <unordered_map>
#include <unordered_set>

#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace symbol {

IR_API DimExpr SimplifyDimExpr(const DimExpr& dim_expr);

IR_API DimExpr SubstituteDimExpr(
    const DimExpr& dim_expr,
    const std::unordered_map<DimExpr, DimExpr>& pattern_to_replacement);

IR_API int GetDimExprPriority(const DimExpr& dim_expr);

enum class PriorityComparisonStatus {
  HIGHER,  // lhs has a higher priority than rhs
  EQUAL,   // lhs and rhs have equal priority
  LOWER    // lhs has a lower priority than rhs
};
IR_API PriorityComparisonStatus CompareDimExprPriority(const DimExpr& lhs,
                                                       const DimExpr& rhs);

enum class DimExprCompareResult {
  GT,       // lhs is greater than rhs
  EQ,       // lhs and rhs is equal
  LT,       // lhs is less than rhs
  UNKNOWN,  // lhs and rhs is not comparable
};
IR_API DimExprCompareResult Compare(const DimExpr& lhs, const DimExpr& rhs);

IR_API std::unordered_set<std::string> CollectDimExprSymbols(
    const DimExpr& dim_expr);

}  // namespace symbol
