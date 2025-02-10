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

#include <unordered_set>

#include "paddle/common/union_find_set.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace symbol {

class IR_API ConstraintsManager {
 public:
  ConstraintsManager() = default;
  ConstraintsManager(const ConstraintsManager&) = delete;
  ConstraintsManager(ConstraintsManager&&) = delete;

  void AddEqCstr(const DimExpr& lhs, const DimExpr& rhs);

  bool IsEqual(const DimExpr& lhs, const DimExpr& rhs) const;

  void AddGTOneCstr(const DimExpr& dim_expr);

  bool IsGTOne(const DimExpr& dim_expr) const;

  void AddBroadcastableCstr(const DimExpr& lhs, const DimExpr& rhs);

  bool IsBroadcastable(const DimExpr& lhs, const DimExpr& rhs) const;

  struct Range {
    std::int64_t min;
    std::int64_t max;
    // TODO(Hongqing-work): Substitute INT32_MAX with a more meaningful value.
    Range() : min(1), max(INT32_MAX) {}
    Range(int min_val, int max_val) : min(min_val), max(max_val) {}
  };
  void AddInputRangeCstr(const DimExpr& dim_expr, const Range& range);

  bool IsBoundedInput(const DimExpr& dim_expr) const;

  const Range& GetRangeOfBoundedInput(const DimExpr& dim_expr) const;

  using EqualConstraints = common::UnionFindSet<DimExpr>;
  using GTOneConstraints = std::unordered_set<DimExpr>;
  using BroadcastableConstraints = std::unordered_set<Broadcastable<DimExpr>>;
  using InputRangeConstraints = std::unordered_map<DimExpr, Range>;

  void VisitEqualClusters(
      const std::function<void(const std::vector<DimExpr>&)>& DoEachCluster)
      const;

  void EqualConstraintsVisitor(
      const std::function<void(std::unordered_map<DimExpr, DimExpr>::iterator)>&
          DoEach);

  void GTOneConstraintsVisitor(
      const std::function<void(GTOneConstraints::iterator)>& DoEach);

  void GTOneConstraintsVisitor(
      const std::function<void(GTOneConstraints::const_iterator)>& DoEach)
      const;

  void BroadcastableConstraintsVisitor(
      const std::function<void(BroadcastableConstraints::iterator)>& DoEach);

  void BroadcastableConstraintsVisitor(
      const std::function<void(BroadcastableConstraints::const_iterator)>&
          DoEach) const;

  void InputRangeConstraintsVisitor(
      const std::function<void(InputRangeConstraints::iterator)>& DoEach);

  void InputRangeConstraintsVisitor(
      const std::function<void(InputRangeConstraints::const_iterator)>& DoEach)
      const;

  using EqualCallbackFunc = std::function<void(const DimExpr&, const DimExpr&)>;
  void SetEqualCallbackFunc(EqualCallbackFunc equal_callback_func);

  const EqualConstraints& equals() const { return equals_; }

  const GTOneConstraints& gtones() const { return gtones_; }

  const BroadcastableConstraints& broadcastables() const {
    return broadcastables_;
  }

  const InputRangeConstraints& input_ranges() const { return input_ranges_; }

 private:
  void SubstituteInConstraint(const DimExpr& lhs, const DimExpr& rhs);

 private:
  EqualCallbackFunc equal_callback_func_ = nullptr;

  EqualConstraints equals_;
  GTOneConstraints gtones_;
  BroadcastableConstraints broadcastables_;
  InputRangeConstraints input_ranges_;
};

std::ostream& operator<<(std::ostream& os,
                         const ConstraintsManager& constraints_manager);

}  // namespace symbol
