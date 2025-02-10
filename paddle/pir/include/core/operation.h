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

#include <ostream>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/common/macros.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/ir_mapping.h"
#include "paddle/pir/include/core/iterator.h"
#include "paddle/pir/include/core/op_info.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/visitors.h"
namespace pir {
class OpBase;
class Program;
class OpOperand;
class OpResult;

namespace detail {
class OpResultImpl;
class OpOperandImpl;
}  // namespace detail

class CloneOptions {
 public:
  CloneOptions()
      : clone_regions_{false},
        clone_operands_{false},
        clone_successors_{false} {}
  CloneOptions(bool clone_regions, bool clone_operands, bool clone_successors)
      : clone_regions_(clone_regions),
        clone_operands_(clone_operands),
        clone_successors_(clone_successors) {}

  bool IsCloneRegions() const { return clone_regions_; }
  bool IsCloneOperands() const { return clone_operands_; }
  bool IsCloneSuccessors() const { return clone_successors_; }

  static CloneOptions &All() {
    static CloneOptions all{true, true, true};
    return all;
  }

 private:
  bool clone_regions_{true};
  bool clone_operands_{true};
  bool clone_successors_{true};
};

class IR_API alignas(8) Operation final
    : public DoubleLevelContainer<Operation> {
 public:
  ///
  /// \brief Malloc memory and construct objects in the following order:
  /// OpResultImpls|Operation|OpOperandImpls.
  /// NOTE: Similar to new and delete, the destroy() and the create() need to be
  /// used in conjunction.
  ///
  static Operation *Create(const std::vector<pir::Value> &inputs,
                           const AttributeMap &attributes,
                           const std::vector<pir::Type> &output_types,
                           pir::OpInfo op_info,
                           size_t num_regions = 0,
                           const std::vector<Block *> &successors = {},
                           bool verify = true);
  static Operation *Create(OperationArgument &&op_argument);

  ///
  /// \brief Deep copy all information and create a new operation.
  ///
  Operation *Clone(IrMapping &ir_mapping,
                   CloneOptions options = CloneOptions()) const;
  ///
  /// \brief Destroy the operation objects and free memory by create().
  ///
  void Destroy();

  IrContext *ir_context() const;

  Dialect *dialect() const;

  bool operator==(const Operation &other) const { return this == &other; }

  ///
  /// \brief op attribute related public interfaces
  ///
  const AttributeMap &attributes() const { return attributes_; }
  // return nullptr if attribute not found.
  Attribute attribute(const std::string &key) const {
    auto iter = attributes_.find(key);
    return iter == attributes_.end() ? nullptr : iter->second;
  }

  template <typename T>
  T attribute(const std::string &key) const {
    return attribute(key).dyn_cast<T>();
  }
  void set_attribute(const std::string &key, Attribute value) {
    attributes_[key] = value;
  }
  void erase_attribute(const std::string &key) { attributes_.erase(key); }
  bool HasAttribute(const std::string &key) const {
    return attributes_.find(key) != attributes_.end();
  }

  void set_value_property(const std::string &key,
                          const Property &value,
                          size_t index);

  void *value_property(const std::string &key, size_t index) const;

  ///
  /// \brief op output related public interfaces
  ///
  uint32_t num_results() const { return num_results_; }
  Value result(uint32_t index) const { return OpResult(op_result_impl(index)); }
  template <typename T = Type>
  T result_type(uint32_t index) const {
    return result(index).type().dyn_cast<T>();
  }
  std::vector<Value> results() const;

  ///
  /// \brief op input related public interfaces
  ///
  uint32_t num_operands() const { return num_operands_; }
  OpOperand operand(uint32_t index) const { return op_operand_impl(index); }
  std::vector<OpOperand> operands() const;
  Value operand_source(uint32_t index) const;
  std::vector<Value> operands_source() const;
  Type operand_type(uint32_t index) const { return operand(index).type(); }

  ///
  /// \brief op successor related public interfaces
  ///
  uint32_t num_successors() const { return num_successors_; }
  BlockOperand block_operand(uint32_t index) const;
  Block *successor(uint32_t index) const;
  void set_successor(Block *block, unsigned index);
  bool HasSuccessors() { return num_successors_ != 0; }

  ///
  /// \brief region related public interfaces
  ///
  using Element = Region;
  using Iterator = Region *;
  using ConstIterator = const Region *;
  uint32_t num_regions() const { return num_regions_; }
  Region &region(unsigned index);
  const Region &region(unsigned index) const;
  ConstIterator begin() const { return regions_; }
  ConstIterator end() const { return regions_ + num_regions_; }
  Iterator begin() { return regions_; }
  Iterator end() { return regions_ + num_regions_; }

  /// \brief block related public interfaces
  using BlockContainer = DoubleLevelContainer<Operation>;
  BlockContainer &blocks() { return *this; }

  ///
  /// \brief parent related public interfaces
  ///
  Block *GetParent() const { return parent_; }
  Region *GetParentRegion() const;
  Operation *GetParentOp() const;
  Program *GetParentProgram();
  operator Block::Iterator() { return position_; }
  operator Block::ConstIterator() const { return position_; }
  void MoveTo(Block *block, Block::Iterator position);

  void Print(std::ostream &os) const;
  pir::OpInfo info() const { return info_; }
  std::string name() const;

  ///
  /// \brief Operation Walkers
  ///
  template <WalkOrder Order = WalkOrder::PostOrder, typename FuncT>
  void Walk(FuncT &&callback) {
    return pir::Walk<Order>(this, std::forward<FuncT>(callback));
  }

  ///
  /// \brief Remove this operation from its parent block and delete it.
  ///
  void Erase();

  ///
  /// \brief Returns true if this operation has no uses.
  ///
  bool use_empty();

  template <typename T>
  T dyn_cast() const {
    return CastUtil<T>::call(this);
  }

  template <typename T>
  bool isa() const {
    return T::classof(this);
  }

  template <typename Trait>
  bool HasTrait() const {
    return info_.HasTrait<Trait>();
  }

  template <typename Interface>
  bool HasInterface() const {
    return info_.HasInterface<Interface>();
  }

  /// Replace all uses of results of this operation with the provided 'values'.
  void ReplaceAllUsesWith(const std::vector<Value> &values);

  void ReplaceAllUsesWith(const std::vector<OpResult> &op_results);

  inline void ReplaceAllUsesWith(Value value) {
    ReplaceAllUsesWith(std::vector<Value>{value});
  }

  void Verify();

  uint64_t id() const { return id_; }

 private:
  DISABLE_COPY_AND_ASSIGN(Operation);
  Operation(const AttributeMap &attribute,
            pir::OpInfo op_info,
            uint32_t num_results,
            uint32_t num_operands,
            uint32_t num_regions,
            uint32_t num_successors);

  int32_t ComputeOpResultOffset(uint32_t index) const;
  detail::OpResultImpl *op_result_impl(uint32_t index) const;

  int32_t ComputeOpOperandOffset(uint32_t index) const;
  detail::OpOperandImpl *op_operand_impl(uint32_t index) const;

  template <typename To, typename Enabler = void>
  struct CastUtil {
    static To call(const Operation *op) {
      throw("Can't dyn_cast to To, To should be a Op or Trait or Interface");
    }
  };

  // Allow access to 'SetParent'.
  friend class Block;
  void SetParent(Block *parent, const Block::Iterator &position);

  template <typename To>
  struct CastUtil<
      To,
      typename std::enable_if<std::is_base_of<OpBase, To>::value>::type> {
    static To call(const Operation *op) { return To::dyn_cast(op); }
  };

  AttributeMap attributes_;

  // store data that user create by Python
  std::vector<PropertyMap> value_properties_;

  OpInfo info_;

  static uint64_t GenerateId() {
    static std::atomic<std::uint64_t> uid{0};
    return ++uid;
  }

  const uint32_t num_results_ = 0;
  const uint32_t num_operands_ = 0;
  const uint32_t num_regions_ = 0;
  const uint32_t num_successors_ = 0;
  const uint64_t id_ = -1;

  detail::BlockOperandImpl *block_operands_{nullptr};
  Region *regions_{nullptr};
  Block *parent_{nullptr};
  Block::Iterator position_;
};

IR_API std::ostream &operator<<(std::ostream &os, const Operation &op);

}  // namespace pir
