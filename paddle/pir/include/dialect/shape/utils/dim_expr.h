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

#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

#include "glog/logging.h"
#include "paddle/common/enforce.h"
#include "paddle/common/overloaded.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/utils.h"

namespace symbol {

#define SYMBOL_NOT_IMPLEMENTED \
  PADDLE_THROW(common::errors::Unimplemented("Not Implemented"))

template <typename T>
struct UnaryDimExpr {
  explicit UnaryDimExpr(const T& d) : data(std::make_shared<Data>(d)) {}
  struct Data {
    explicit Data(const T& d) : data(d) {}
    T data;
  };

  const Data& operator*() const { return *data; }
  Data& operator*() { return *data; }
  const Data* operator->() const { return data.get(); }
  Data* operator->() { return data.get(); }

  std::shared_ptr<Data> data;
};

template <typename T>
struct BinaryDimExpr {
  explicit BinaryDimExpr(const T& l, const T& r)
      : data(std::make_shared<Data>(l, r)) {}

  struct Data {
    explicit Data(const T& l, const T& r) : lhs(l), rhs(r) {}
    T lhs;
    T rhs;
  };

  const Data& operator*() const { return *data; }
  Data& operator*() { return *data; }
  const Data* operator->() const { return data.get(); }
  Data* operator->() { return data.get(); }

  std::shared_ptr<Data> data;
};

template <typename T>
class List final {
 public:
  List(const List&) = default;
  List(List&&) = default;
  List& operator=(const List&) = default;
  List& operator=(List&&) = default;

  using value_type = T;

  explicit List() : vector_(std::make_shared<std::vector<T>>()) {}

  template <
      typename Arg,
      std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, List>, bool> = true>
  explicit List(Arg&& arg)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector<T>{std::forward<Arg>(arg)})) {}

  template <typename Arg0, typename Arg1, typename... Args>
  List(Arg0&& arg0, Arg1&& arg1, Args&&... args)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector<T>{std::forward<Arg0>(arg0),
                           std::forward<Arg1>(arg1),
                           std::forward<Args>(args)...})) {}

  bool operator==(const List& other) const {
    if (&vector() == &other.vector()) {
      return true;
    }
    return vector() == other.vector();
  }

  bool operator!=(const List& other) const { return !(*this == other); }

  std::vector<T>& operator*() const { return *vector_; }
  std::vector<T>* operator->() const { return vector_.get(); }

  const std::vector<T>& vector() const { return *vector_; }

  const auto& Get(std::size_t idx) const { return vector_->at(idx); }

 private:
  std::shared_ptr<std::vector<T>> vector_;
};

template <typename T>
struct Negative final : public UnaryDimExpr<T> {
  using UnaryDimExpr<T>::UnaryDimExpr;
};

template <typename T>
struct Add final {
  List<T> operands;
};

template <typename T>
struct Mul final {
  List<T> operands;
};

template <typename T>
struct Div final : public BinaryDimExpr<T> {
  using BinaryDimExpr<T>::BinaryDimExpr;
};

template <typename T>
struct Max final {
  List<T> operands;
};

template <typename T>
struct Min final {
  List<T> operands;
};

template <typename T>
struct Broadcast final {
  List<T> operands;
};

template <typename T>
struct Equal final : public BinaryDimExpr<T> {
  using BinaryDimExpr<T>::BinaryDimExpr;
};

template <typename T>
struct Broadcastable final : public BinaryDimExpr<T> {
  using BinaryDimExpr<T>::BinaryDimExpr;
  bool operator==(const Broadcastable& other) const {
    return (this->data->lhs == other.data->lhs &&
            this->data->rhs == other.data->rhs) ||
           (this->data->lhs == other.data->rhs &&
            this->data->rhs == other.data->lhs);
  }
};

class DimExpr;

// DimExpr = std::int64_t
//         | std::string
//         | Negative DimExpr
//         | Add DimExpr
//         | Mul DimExpr
//         | Div DimExpr
//         | Max DimExpr
//         | Min DimExpr
//         | Broadcast DimExpr
using DimExprBase = std::variant<std::int64_t,
                                 std::string,
                                 Negative<DimExpr>,
                                 Add<DimExpr>,
                                 Mul<DimExpr>,
                                 Div<DimExpr>,
                                 Max<DimExpr>,
                                 Min<DimExpr>,
                                 Broadcast<DimExpr>>;

class IR_API DimExpr : public DimExprBase {
 public:
  using DimExprBase::DimExprBase;

  template <typename T>
  bool isa() const {
    return std::holds_alternative<T>(*this);
  }

  template <typename T>
  const T& dyn_cast() const {
    return std::get<T>(*this);
  }

  template <typename T>
  bool Has() const {
    return std::holds_alternative<T>(*this);
  }

  template <typename T>
  const T& Get() const {
    return std::get<T>(*this);
  }

  const DimExprBase& variant() const {
    return static_cast<const DimExprBase&>(*this);
  }

  DEFINE_MATCH_METHOD();

  DimExpr operator+(const DimExpr& other) const;
  DimExpr operator-(const DimExpr& other) const;
  DimExpr operator*(const DimExpr& other) const;
  DimExpr operator/(const DimExpr& other) const;
  bool operator==(const DimExpr& other) const;
  bool operator!=(const DimExpr& other) const;
};

// DimExprConstraint = Equal DimExpr
//                   | Broadcastable DimExpr
using DimExprConstraint = std::variant<Equal<DimExpr>, Broadcastable<DimExpr>>;

IR_API std::string ToString(const DimExpr& dim_expr);

IR_API std::ostream& operator<<(std::ostream&, const DimExpr& dim_expr);

IR_API std::ostream& operator<<(std::ostream&,
                                const std::vector<DimExpr>& dim_exprs);

IR_API std::size_t GetHashValue(const DimExpr& dim_expr);

}  // namespace symbol

namespace std {

template <>
struct hash<symbol::DimExpr> {
  std::size_t operator()(const symbol::DimExpr& dim_expr) const {
    return symbol::GetHashValue(dim_expr);
  }
};

template <>
struct hash<std::vector<symbol::DimExpr>> {
  std::size_t operator()(const std::vector<symbol::DimExpr>& dim_exprs) const {
    std::size_t hash_value = 0;
    const auto hash_func = std::hash<symbol::DimExpr>();
    for (const auto& dim_expr : dim_exprs) {
      hash_value = pir::detail::hash_combine(hash_value, hash_func(dim_expr));
    }
    return hash_value;
  }
};

template <>
struct hash<symbol::Broadcastable<symbol::DimExpr>> {
  std::size_t operator()(
      const symbol::Broadcastable<symbol::DimExpr>& broadcastable) const {
    return pir::detail::hash_combine(GetHashValue(broadcastable.data->lhs),
                                     GetHashValue(broadcastable.data->rhs));
  }
};

}  // namespace std
