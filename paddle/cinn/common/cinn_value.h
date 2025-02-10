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
#include <absl/types/any.h>
#include <glog/logging.h>

#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/common/enforce.h"
struct cinn_buffer_t;

namespace cinn {

namespace ir {

class Expr;
class Var;
class LoweredFunc;

}  // namespace ir

namespace common {

template <typename T>
cinn_value_t ToValue(T v);

class CINNValue;
class CINNValuePack;

/**
 * A _CINNValuePack_ is a shared Array of multiple CINNValue.
 */
struct _CINNValuePack_ : public cinn::common::Object {
  /**
   * Create a new CINNValuePack instance.
   * @param array The list of CINNValues.
   * @return a CINNValuePack.
   */
  static CINNValuePack Make(const std::vector<CINNValue>& array);

  //! Get i-th element in mutable mode.
  CINNValue& operator[](int offset);
  //! Get i-th element in readonly mode.
  const CINNValue& operator[](int offset) const;

  //! Add one \p value to the tail.
  void AddValue(const CINNValue& value);

  //! Remove all the values.
  void Clear();

  size_t size() const { return values_.size(); }

  bool empty() const { return values_.empty(); }

  CINN_DISALLOW_COPY_AND_ASSIGN(_CINNValuePack_);

  const char* type_info() const override;

 private:
  _CINNValuePack_() = default;
  std::vector<CINNValue> values_;
  static constexpr char* __type_info__ = "CINNValuePack";
};

struct CINNValuePack : public Shared<_CINNValuePack_> {
  explicit CINNValuePack(_CINNValuePack_* ptr) : Shared<_CINNValuePack_>(ptr) {}
  explicit CINNValuePack(const std::vector<CINNValue>& array)
      : Shared<_CINNValuePack_>(_CINNValuePack_::Make(array)) {}
  CINNValue& operator[](int offset) { return (*operator->())[offset]; }
  const CINNValue& operator[](int offset) const {
    return (*operator->())[offset];
  }

  size_t size() const { return (*operator->()).size(); }

  bool empty() const { return (*operator->()).empty(); }

  CINNValue& back() {
    PADDLE_ENFORCE_GT((*operator->()).size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The size of the array should greater than 0."));
    return (*operator->())[size() - 1];
  }

  const CINNValue& back() const {
    PADDLE_ENFORCE_GT((*operator->()).size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The size of the array should greater than 0."));
    return (*operator->())[size() - 1];
  }

  _CINNValuePack_* operator->() { return get(); }
  const _CINNValuePack_* operator->() const { return get(); }
};

/**
 * Handler for value types in CINN system. It supports two kinds of values: the
 * POD and Shared.
 */
class CINNValue : public cinn_pod_value_t {
 public:
  static constexpr int kNull = -1;

  CINNValue() : cinn_pod_value_t(cinn_value_t(), kNull) {}
  CINNValue(cinn_value_t value, int type_code)
      : cinn_pod_value_t(value, type_code) {}

  explicit CINNValue(bool value) : cinn_pod_value_t(value) {
    type_code_ = ::cinn_type_code<bool>();
  }
  explicit CINNValue(int32_t value) : cinn_pod_value_t(value) {
    type_code_ = ::cinn_type_code<int32_t>();
  }
  explicit CINNValue(int64_t value) : cinn_pod_value_t(value) {
    type_code_ = ::cinn_type_code<int64_t>();
  }
  explicit CINNValue(float value) : cinn_pod_value_t(value) {
    type_code_ = ::cinn_type_code<float>();
  }

  explicit CINNValue(double value) : cinn_pod_value_t(value) {
    type_code_ = ::cinn_type_code<double>();
  }
  explicit CINNValue(char* value);
  explicit CINNValue(cinn_buffer_t* value) : cinn_pod_value_t(value) {}
  explicit CINNValue(void* value) : cinn_pod_value_t(value) {}
  explicit CINNValue(const char* value) : cinn_pod_value_t(value) {}
  explicit CINNValue(const std::string&);
  explicit CINNValue(const ir::Var& value);
  explicit CINNValue(const ir::Expr& value);
  explicit CINNValue(const ir::LoweredFunc& value);
  explicit CINNValue(const CINNValuePack& value);

  bool defined() const { return type_code_ != kNull; }

  //! The value getters for the supported types.
  // @{
  using cinn_pod_value_t::operator double;
  using cinn_pod_value_t::operator float;
  using cinn_pod_value_t::operator bool;
  using cinn_pod_value_t::operator int32_t;
  using cinn_pod_value_t::operator int64_t;
  using cinn_pod_value_t::operator void*;
  using cinn_pod_value_t::operator cinn_buffer_t*;
  using cinn_pod_value_t::operator char*;
  operator std::string() const;
  operator ir::Var() const;
  operator ir::Expr() const;
  operator CINNValuePack() const;
  // @}

  bool is_string() const;
  bool is_var() const;
  bool is_expr() const;
  bool is_tensor() const;

  //! Assign operators
  // @{
  CINNValue& operator=(bool value);
  CINNValue& operator=(int32_t value);
  CINNValue& operator=(int64_t value);
  CINNValue& operator=(float value);
  CINNValue& operator=(double value);
  CINNValue& operator=(bfloat16 value);
  CINNValue& operator=(float16 value);
  CINNValue& operator=(char* value);
  CINNValue& operator=(const std::string& value);
  CINNValue& operator=(const ir::Var& value);
  CINNValue& operator=(const ir::Expr& value);
  CINNValue& operator=(cinn_buffer_t* value);
  CINNValue& operator=(void* value);
  CINNValue& operator=(const CINNValuePack& value);
  CINNValue& operator=(const char* value);
  // @}

  //  //! Set the value.
  //  template <typename T>
  //  void Set(T v) {
  //    if constexpr (std::is_same_v<std::decay_t<T>, CINNValue>) {
  //      *this = v;
  //    } else {
  //      *this = CINNValue(v);
  //    }
  //  }

  template <typename T>
  inline void _Set(T v, std::true_type) {
    *this = v;
  }

  template <typename T>
  inline void _Set(T v, std::false_type) {
    *this = CINNValue(v);
  }
  // using tag-dispatch instead of constexpr if
  template <typename T>
  void Set(T v) {
    _Set(v, std::is_same<std::decay_t<T>, CINNValue>{});
  }

  /**
   * Get the type code for a specific POD type.
   * @param T some data type.
   * @return an integer representing the type code.
   */
  template <typename T>
  static int TypeCode();

 protected:
  absl::any shared_;
};

}  // namespace common
}  // namespace cinn
