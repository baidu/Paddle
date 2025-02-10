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

#include <type_traits>
#include "paddle/cinn/adt/dim_expr.h"
#include "paddle/cinn/adt/match.h"
#include "paddle/cinn/adt/print_utils/print_dim_expr.h"

namespace cinn::adt {

template <template <typename> class Op, typename T0>
struct UnaryDimExprMatchTrait {
  using base_type = Op<DimExpr>;

  static constexpr int is_template = true;

  template <template <typename, typename> class Matcher>
  static bool MatchChildren(const base_type& value) {
    return Matcher<T0, DimExpr>::Call(std::get<0>(value.tuple()));
  }
};

template <template <typename> class Op, typename T0>
struct BinaryDimExprMatchTrait {
  using base_type = Op<DimExpr>;

  static constexpr int is_template = true;

  template <template <typename, typename> class Matcher>
  static bool MatchChildren(const base_type& value) {
    const auto& lhs = value->lhs;
    const auto& rhs = value->rhs;
    return Matcher<T0, DimExpr>::Call(lhs) && Matcher<T0, DimExpr>::Call(rhs);
  }
};

template <template <typename> class Op, typename T0>
struct ListDimExprMatchTrait {
  using base_type = Op<DimExpr>;

  static constexpr int is_template = true;

  template <template <typename, typename> class Matcher>
  static bool MatchChildren(const base_type& value) {
    const auto& [operands] = value;
    for (const auto& operand : *operands) {
      if (!Matcher<T0, DimExpr>::Call(operand)) {
        return false;
      }
    }
    return true;
  }
};

template <>
struct MatchTrait<DimExpr, std::int64_t> final {
  static constexpr int is_template = false;
};

template <>
struct MatchTrait<DimExpr, SymbolicDim> final {
  static constexpr int is_template = false;
};

template <typename T0>
struct MatchTrait<DimExpr, ::symbol::Negative<T0>> final
    : public UnaryDimExprMatchTrait<::symbol::Negative, T0> {};

template <typename T0>
struct MatchTrait<DimExpr, ::symbol::Add<T0>> final
    : public ListDimExprMatchTrait<::symbol::Add, T0> {};

template <typename T0>
struct MatchTrait<DimExpr, ::symbol::Mul<T0>> final
    : public ListDimExprMatchTrait<::symbol::Mul, T0> {};

template <typename T0>
struct MatchTrait<DimExpr, ::symbol::Div<T0>> final
    : public BinaryDimExprMatchTrait<::symbol::Div, T0> {};

template <typename T0>
struct MatchTrait<DimExpr, ::symbol::Broadcast<T0>> final
    : public ListDimExprMatchTrait<::symbol::Broadcast, T0> {};

}  // namespace cinn::adt
