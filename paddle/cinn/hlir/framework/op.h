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
#include <absl/container/flat_hash_map.h>
#include <absl/types/any.h>
#include <glog/logging.h>

#include <atomic>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>  //NOLINT
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/utils/registry.h"
#include "paddle/cinn/utils/type_defs.h"
#include "paddle/common/enforce.h"
template <typename R, typename... Args>
inline auto MakeOpFunction(R (*func)(Args...)) {
  return std::function<R(Args...)>(func);
}

namespace cinn {
namespace hlir {
namespace framework {
class Operator;

using shape_t = utils::ShapeType;
using dim_t = utils::DimType;

/*! \brief operator pattern used in graph fusion */
enum OpPatternKind {
  // The relation between input tensor index and output tensor index is
  // one-to-one correspondence.
  // for example :code:`out[i, j] = input[i, j] + 1`.
  // Note that the axis need to be in order.
  kElementWise = 0,
  // The relation between input tensor index and output tensor index is
  // one-to-many correspondence.
  // for example :code:`out[i, j, k] = input[i, j]`.
  // Note that the axis need to be in order.
  kBroadcast = 1,
  // Injective operator, we can always injectively map a output axis to a input
  // axis.
  // for example :code:`out[i, j] = input[j, i]`.
  kInjective = 2,
  // The relation between input tensor index and output tensor index is
  // many-to-one correspondence.
  // for example :code:`out[i, j] = sum(input[i, j, k]) along k`.
  kReduction = 3,
  // Complex operation, can still fuse one-to-one operations into its output.
  kOutFusible = 4,
  // Operation that cannot fuse anything.
  kNonFusible = 8
};

struct OpRegistry : public Registry<Operator> {
  std::recursive_mutex mutex;
  std::atomic<int> op_counter{0};
  absl::flat_hash_map<std::string, std::unique_ptr<absl::any>> attrs;

  static OpRegistry* Global() {
    static OpRegistry x;
    return &x;
  }

 private:
  OpRegistry() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(OpRegistry);
};

template <typename ValueType>
class OpValueType {
 public:
  inline const ValueType& operator[](const Operator* op) const;

  inline const ValueType& Get(const Operator* op,
                              const ValueType& def_value) const;

  inline bool Find(const Operator* op) const;

  size_t Size() const { return data.size(); }

 private:
  friend class Operator;
  std::string attr_name;
  std::vector<ValueType> data;
  OpValueType() = default;
};

class Operator {
 public:
  std::string name;
  std::string description;
  uint32_t num_inputs{1};
  uint32_t num_outputs{1};
  uint32_t support_level{10};

  inline Operator& describe(const std::string description) {
    this->description = description;
    return *this;
  }

  inline Operator& set_num_inputs(uint32_t n) {
    this->num_inputs = n;
    return *this;
  }

  inline Operator& set_num_outputs(uint32_t n) {
    this->num_outputs = n;
    return *this;
  }

  inline Operator& set_support_level(uint32_t n) {
    this->support_level = n;
    return *this;
  }
  /**
   * \brief Get an Op for a given operator name.
   *  Will raise an error if the op has not been registered.
   * @param op_name Name of the operator.
   * @return Pointer to a Op, valid throughout program lifetime.
   */
  static const Operator* Get(const std::string& op_name) {
    const Operator* op = OpRegistry::Global()->Find(op_name);
    PADDLE_ENFORCE_NOT_NULL(op,
                            ::common::errors::PreconditionNotMet(
                                "Operator [%s] is not registered", op_name));
    return op;
  }

  template <typename ValueType>
  inline Operator& set_attr(const std::string& attr_name,
                            const ValueType& value) {
    static std::ofstream static_outfile;
    static_outfile.open("/ssd2/liuchao52/Paddle/liuchao/static.txt",
                        std::ios::out | std::ios::trunc);
    static std::ofstream dynamic_outfile;
    dynamic_outfile.open("/ssd2/liuchao52/Paddle/liuchao/dynamic.txt",
                         std::ios::out | std::ios::trunc);

    if (attr_name == "CINNStrategy") {
      VLOG(-1) << "static_op: " << name;
      static_outfile << name << ", ";
    }
    if (attr_name == "CINNStrategySymbolic") {
      VLOG(-1) << "dynamic_op: " << name;
      dynamic_outfile << name << ", ";
    }
    UpdateAttrMap(attr_name, [this, attr_name, value](absl::any* pmap) {
      if (!pmap->has_value()) {
        OpValueType<ValueType> pm;
        pm.attr_name = attr_name;
        *pmap = std::move(pm);
      }
      std::vector<ValueType>& vec =
          absl::any_cast<OpValueType<ValueType>&>(*pmap).data;
      // resize the value type.
      if (vec.size() <= index) {
        vec.resize(index + 1, ValueType());
      }
      vec[index] = value;
    });
    return *this;
  }
  template <typename ValueType>
  static const OpValueType<ValueType>& GetAttrs(const std::string& attr_name) {
    const absl::any* ref = GetAttrMap(attr_name);
    if (ref == nullptr) {
      //! update the attribute map of the key by creating new empty OpMap
      UpdateAttrMap(attr_name, [attr_name](absl::any* pmap) {
        if (!pmap->has_value()) {
          OpValueType<ValueType> pm;
          pm.attr_name = attr_name;
          *pmap = std::move(pm);
        }
      });
      ref = GetAttrMap(attr_name);
    }
    return absl::any_cast<const OpValueType<ValueType>&>(*ref);
  }

  auto get_index() const { return index; }

 private:
  template <typename ValueType>
  friend class OpValueType;
  friend class Registry<Operator>;
  uint32_t index{0};
  Operator() { index = OpRegistry::Global()->op_counter++; }
  static const absl::any* GetAttrMap(const std::string& key) {
    auto& dict = OpRegistry::Global()->attrs;
    auto it = dict.find(key);
    if (it != dict.end()) {
      return it->second.get();
    } else {
      return nullptr;
    }
  }
  //! update the attribute OpValueType
  static void UpdateAttrMap(const std::string& key,
                            std::function<void(absl::any*)> updater) {
    OpRegistry* reg = OpRegistry::Global();
    std::lock_guard<std::recursive_mutex>(reg->mutex);
    std::unique_ptr<absl::any>& value = reg->attrs[key];
    if (value.get() == nullptr) value.reset(new absl::any());
    if (updater != nullptr) updater(value.get());
  }
};

template <typename ValueType>
const ValueType& OpValueType<ValueType>::operator[](const Operator* op) const {
  PADDLE_ENFORCE_NOT_NULL(
      op,
      ::common::errors::PreconditionNotMet(
          "The input op is nullptr and it is invalid! Please check again."));
  const uint32_t idx = op->index;
  PADDLE_ENFORCE_LT(idx,
                    data.size(),
                    ::common::errors::InvalidArgument(
                        "Attribute  has not been registered for Operator"));
  return data[idx];
}

template <typename ValueType>
const ValueType& OpValueType<ValueType>::Get(const Operator* op,
                                             const ValueType& def_value) const {
  if (!op) return def_value;
  const uint32_t idx = op->index;
  if (idx < data.size()) {
    return data[idx];
  } else {
    return def_value;
  }
}

template <typename ValueType>
bool OpValueType<ValueType>::Find(const Operator* op) const {
  if (!op) return false;
  const uint32_t idx = op->index;
  return idx < data.size();
}

// internal macros to make
#define CINN_REGISTER_VAR_DEF(OpName) \
  static ::cinn::hlir::framework::Operator& __make_##HlirOp##_##OpName

/**
 * @def CINN_REGISTER_OP
 * \brief Register a new operator, or set attribute of the corresponding op.
 *
 * @param OpName The name of registry
 *
 * \code
 *  CINN_REGISTER_OP(add)
 *  .describe("add two inputs together")
 *  .set_num_inputs(2)
 *  .set_attr<OpKernel>("gpu_kernel", AddKernel);
 * \endcode
 */
#define CINN_REGISTER_OP(OpName)                                          \
  CINN_STR_CONCAT(CINN_REGISTER_VAR_DEF(OpName), __COUNTER__) =           \
      ::cinn::hlir::framework::OpRegistry::Global()->__REGISTER_OR_GET__( \
          #OpName)

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
