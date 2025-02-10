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

#include "paddle/pir/include/core/value.h"
#include "paddle/pir/src/core/op_operand_impl.h"

#define OUTLINE_RESULT_IDX 6u
#define MAX_INLINE_RESULT_IDX (OUTLINE_RESULT_IDX - 1u)
#define BLOCK_ARG_IDX (OUTLINE_RESULT_IDX + 1u)

namespace pir {
class Operation;

namespace detail {
///
/// \brief ValueImpl is the base class of all derived Value classes such as
/// OpResultImpl. This class defines all the information and usage interface in
/// the IR Value. Each Value include three attributes:
/// (1) type: pir::Type; (2) UD-chain of value: OpOperandImpl*, first op_operand
/// address with offset of this value; (3) index: the position where the output
/// list of the parent operator.
///
class alignas(8) ValueImpl {
 public:
  ///
  /// \brief Interface functions of "type_" attribute.
  ///
  Type type() const { return type_; }

  void set_type(Type type) { type_ = type; }

  OpOperandImpl *first_use() const {
    return reinterpret_cast<OpOperandImpl *>(
        reinterpret_cast<uintptr_t>(first_use_offsetted_by_kind_) & (~0x07));
  }

  void set_first_use(OpOperandImpl *first_use);

  OpOperandImpl **first_use_addr() { return &first_use_offsetted_by_kind_; }

  bool use_empty() const { return first_use() == nullptr; }

  bool HasOneUse() const {
    return (first_use() != nullptr) && (first_use()->next_use() == nullptr);
  }

  std::string PrintUdChain();

  ///
  /// \brief Interface functions of "first_use_offsetted_by_kind_" attribute.
  ///
  uint32_t kind() const {
    return reinterpret_cast<uintptr_t>(first_use_offsetted_by_kind_) & 0x07;
  }

  template <typename T>
  bool isa() const {
    return T::classof(*this);
  }

  uint64_t id() const { return id_; }

 protected:
  ///
  /// \brief Only can be constructed by derived classes such as OpResultImpl.
  ///
  ValueImpl(Type type, uint32_t kind);

  ///
  /// \brief Attribute1: Type of value.
  ///
  Type type_;

  ///
  /// \brief Attribute2/3: Record the UD-chain of value and index.
  /// NOTE: The members of the OpOperandImpl include four pointers, so this
  /// class is 8-byte aligned, and the lower 3 bits of its address are 0, so the
  /// index can be stored in these 3 bits, stipulate:
  /// (1) index = 0~5: represent positions 0 to 5 inline
  /// output(OpInlineResultImpl); (2) index = 6: represent the position >=6
  /// outline output(OpOutlineResultImpl); (3) index = 7 is reserved.
  ///
  OpOperandImpl *first_use_offsetted_by_kind_ = nullptr;

  const uint64_t id_ = 0;
};

}  // namespace detail

}  // namespace pir
