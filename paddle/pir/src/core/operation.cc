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

#include <glog/logging.h>
#include <cstdint>
#include <ostream>

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/op_info.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/src/core/block_operand_impl.h"
#include "paddle/pir/src/core/op_result_impl.h"

namespace pir {
using detail::OpInlineResultImpl;
using detail::OpOperandImpl;
using detail::OpOutlineResultImpl;
using detail::OpResultImpl;

Operation *Operation::Create(OperationArgument &&argument) {
  Operation *op = Create(argument.inputs,
                         argument.attributes,
                         argument.output_types,
                         argument.info,
                         argument.regions.size(),
                         argument.successors);
  for (size_t i = 0; i < argument.regions.size(); ++i) {
    if (argument.regions[i]) {
      op->region(i).TakeBody(std::move(*argument.regions[i]));
    }
  }
  return op;
}

// Allocate the required memory based on the size and number of inputs, outputs,
// and operators, and construct it in the order of: OpOutlineResult,
// OpInlineResult, Operation, operand.
Operation *Operation::Create(const std::vector<Value> &inputs,
                             const AttributeMap &attributes,
                             const std::vector<Type> &output_types,
                             pir::OpInfo op_info,
                             size_t num_regions,
                             const std::vector<Block *> &successors,
                             bool verify) {
  // 1. Calculate the required memory size for OpResults + Operation +
  // OpOperands.
  uint32_t num_results = output_types.size();
  uint32_t num_operands = inputs.size();
  uint32_t num_successors = successors.size();
  uint32_t max_inline_result_num = MAX_INLINE_RESULT_IDX + 1;
  size_t result_mem_size =
      num_results > max_inline_result_num
          ? sizeof(detail::OpOutlineResultImpl) *
                    (num_results - max_inline_result_num) +
                sizeof(detail::OpInlineResultImpl) * max_inline_result_num
          : sizeof(detail::OpInlineResultImpl) * num_results;
  size_t op_mem_size = sizeof(Operation);
  size_t operand_mem_size = sizeof(detail::OpOperandImpl) * num_operands;
  size_t block_operand_size = num_successors * sizeof(detail::BlockOperandImpl);
  size_t region_mem_size = num_regions * sizeof(Region);
  size_t base_size = result_mem_size + op_mem_size + operand_mem_size +
                     region_mem_size + block_operand_size;
  // 2. Malloc memory.
  char *base_ptr =
      reinterpret_cast<char *>(detail::aligned_malloc(base_size, 8));

  auto name = op_info ? op_info.name() : "";
  VLOG(10) << "Create Operation [" << name
           << "]: {ptr = " << static_cast<void *>(base_ptr)
           << ", size = " << base_size << "} done.";
  // 3.1. Construct OpResults.
  for (size_t idx = num_results; idx > 0; idx--) {
    if (idx > max_inline_result_num) {
      new (base_ptr)
          detail::OpOutlineResultImpl(output_types[idx - 1], idx - 1);
      base_ptr += sizeof(detail::OpOutlineResultImpl);
    } else {
      new (base_ptr) detail::OpInlineResultImpl(output_types[idx - 1], idx - 1);
      base_ptr += sizeof(detail::OpInlineResultImpl);
    }
  }
  // 3.2. Construct Operation.
  Operation *op = new (base_ptr) Operation(attributes,
                                           op_info,
                                           num_results,
                                           num_operands,
                                           num_regions,
                                           num_successors);
  base_ptr += sizeof(Operation);
  // 3.3. Construct OpOperands.
  if ((reinterpret_cast<uintptr_t>(base_ptr) & 0x7) != 0) {
    IR_THROW("The address of OpOperandImpl must be divisible by 8.");
  }
  for (size_t idx = 0; idx < num_operands; idx++) {
    new (base_ptr) detail::OpOperandImpl(inputs[idx], op);
    base_ptr += sizeof(detail::OpOperandImpl);
  }
  // 3.4. Construct BlockOperands.
  if (num_successors > 0) {
    op->block_operands_ =
        reinterpret_cast<detail::BlockOperandImpl *>(base_ptr);
    for (size_t idx = 0; idx < num_successors; idx++) {
      new (base_ptr) detail::BlockOperandImpl(successors[idx], op);
      base_ptr += sizeof(detail::BlockOperandImpl);
    }
  }
  // 3.5. Construct Regions
  if (num_regions > 0) {
    op->regions_ = reinterpret_cast<Region *>(base_ptr);
    for (size_t idx = 0; idx < num_regions; idx++) {
      new (base_ptr) Region(op);
      base_ptr += sizeof(Region);
    }
  }
  // 0. Verify
  if (verify && op_info) {
    try {
      op_info.VerifySig(op);
    } catch (const common::enforce::EnforceNotMet &e) {
      op->Destroy();
      throw e;
    }
  }
  return op;
}

Operation *Operation::Clone(IrMapping &ir_mapping, CloneOptions options) const {
  auto inputs = operands_source();
  if (options.IsCloneOperands()) {
    // replace value by IRMapping inplacely.
    for (auto &value : inputs) {
      value = ir_mapping.Lookup(value);
    }
  }

  std::vector<Type> output_types;
  for (auto &result : results()) {
    output_types.push_back(result.type());
  }

  std::vector<Block *> successors = {};
  if (options.IsCloneSuccessors()) {
    for (uint32_t i = 0; i < num_successors_; ++i) {
      successors.push_back(ir_mapping.Lookup(successor(i)));
    }
  }
  auto *new_op = Create(inputs,
                        attributes_,
                        output_types,
                        info_,
                        num_regions_,
                        successors,
                        false);
  ir_mapping.Add(this, new_op);

  // record outputs mapping info
  for (uint32_t i = 0; i < num_results_; ++i) {
    ir_mapping.Add(result(i), new_op->result(i));
  }

  if (options.IsCloneRegions()) {
    // clone regions recursively
    for (uint32_t i = 0; i < num_regions_; ++i) {
      this->region(i).CloneInto(new_op->region(i), ir_mapping);
    }
  }

  return new_op;
}

// Call destructors for Region , OpResults, Operation, and OpOperands in
// sequence, and finally free memory.
void Operation::Destroy() {
  VLOG(10) << "Destroy Operation [" << name() << "] ...";
  // 1. Deconstruct Regions.
  if (num_regions_ > 0) {
    for (size_t idx = 0; idx < num_regions_; idx++) {
      regions_[idx].~Region();
    }
  }

  // 2. Deconstruct Result.
  for (size_t idx = 0; idx < num_results_; ++idx) {
    detail::ValueImpl *impl = result(idx).impl();
    if (detail::OpOutlineResultImpl::classof(*impl)) {
      static_cast<detail::OpOutlineResultImpl *>(impl)->~OpOutlineResultImpl();
    } else {
      static_cast<detail::OpInlineResultImpl *>(impl)->~OpInlineResultImpl();
    }
  }

  // 3. Deconstruct Properties.
  for (auto &value_property : value_properties_) {
    for (auto &property_map : value_property) {
      if (property_map.second.second) {
        property_map.second.second((property_map.second.first));
      }
    }
  }

  // 4. Deconstruct Operation.
  this->~Operation();

  // 5. Deconstruct OpOperand.
  for (size_t idx = 0; idx < num_operands_; idx++) {
    detail::OpOperandImpl *op_operand_impl = operand(idx).impl_;
    if (op_operand_impl) {
      op_operand_impl->~OpOperandImpl();
    }
  }

  // 6. Deconstruct BlockOperand.
  for (size_t idx = 0; idx < num_successors_; idx++) {
    detail::BlockOperandImpl *block_operand_impl = block_operands_ + idx;
    if (block_operand_impl) {
      block_operand_impl->~BlockOperandImpl();
    }
  }

  // 7. Free memory.
  size_t result_mem_size =
      num_results_ > OUTLINE_RESULT_IDX
          ? sizeof(detail::OpOutlineResultImpl) *
                    (num_results_ - OUTLINE_RESULT_IDX) +
                sizeof(detail::OpInlineResultImpl) * OUTLINE_RESULT_IDX
          : sizeof(detail::OpInlineResultImpl) * num_results_;
  void *aligned_ptr = reinterpret_cast<char *>(this) - result_mem_size;

  VLOG(10) << "Destroy Operation [" << name() << "]: {ptr = " << aligned_ptr
           << ", size = " << result_mem_size << "} done.";
  detail::aligned_free(aligned_ptr);
}

IrContext *Operation::ir_context() const { return info_.ir_context(); }

Dialect *Operation::dialect() const { return info_.dialect(); }

Operation::Operation(const AttributeMap &attributes,
                     pir::OpInfo op_info,
                     uint32_t num_results,
                     uint32_t num_operands,
                     uint32_t num_regions,
                     uint32_t num_successors)
    : attributes_(attributes),
      info_(op_info),
      num_results_(num_results),
      num_operands_(num_operands),
      num_regions_(num_regions),
      num_successors_(num_successors),
      id_(GenerateId()) {}

///
/// \brief op output related public interfaces implementation
///
std::vector<Value> Operation::results() const {
  std::vector<Value> res;
  for (uint32_t i = 0; i < num_results(); ++i) {
    res.push_back(result(i));
  }
  return res;
}

///
/// \brief op input related public interfaces
///
std::vector<OpOperand> Operation::operands() const {
  std::vector<OpOperand> res;
  for (uint32_t i = 0; i < num_operands(); ++i) {
    res.push_back(operand(i));
  }
  return res;
}
Value Operation::operand_source(uint32_t index) const {
  auto val = op_operand_impl(index);
  return val ? val->source() : nullptr;
}

std::vector<Value> Operation::operands_source() const {
  std::vector<Value> res;
  for (uint32_t i = 0; i < num_operands(); ++i) {
    res.push_back(operand_source(i));
  }
  return res;
}

///
/// \brief op successor related public interfaces
///
BlockOperand Operation::block_operand(uint32_t index) const {
  PADDLE_ENFORCE_LT(
      index,
      num_successors_,
      common::errors::InvalidArgument("Invalid block_operand index"));
  return block_operands_ + index;
}
Block *Operation::successor(uint32_t index) const {
  return block_operand(index).source();
}
void Operation::set_successor(Block *block, unsigned index) {
  PADDLE_ENFORCE_LT(
      index,
      num_operands_,
      common::errors::InvalidArgument("Invalid block_operand index"));
  (block_operands_ + index)->set_source(block);
}

///
/// \brief region related public interfaces implementation
///
Region &Operation::region(unsigned index) {
  PADDLE_ENFORCE_LT(index,
                    num_regions_,
                    common::errors::InvalidArgument("invalid region index"));
  return regions_[index];
}
const Region &Operation::region(unsigned index) const {
  PADDLE_ENFORCE_LT(index,
                    num_regions_,
                    common::errors::InvalidArgument("invalid region index"));
  return regions_[index];
}

///
/// \brief parent related public interfaces implementation
///
Region *Operation::GetParentRegion() const {
  return parent_ ? parent_->GetParent() : nullptr;
}
Operation *Operation::GetParentOp() const {
  return parent_ ? parent_->GetParentOp() : nullptr;
}
Program *Operation::GetParentProgram() {
  auto op = this;
  while (Operation *parent_op = op->GetParentOp()) {
    op = parent_op;
  }
  ModuleOp module_op = op->dyn_cast<ModuleOp>();
  return module_op ? module_op.program() : nullptr;
}
void Operation::SetParent(Block *parent, const Block::Iterator &position) {
  parent_ = parent;
  position_ = position;
}

void Operation::MoveTo(Block *block, Block::Iterator position) {
  PADDLE_ENFORCE_NOT_NULL(
      parent_,
      common::errors::InvalidArgument("Operation does not have parent"));
  Operation *op = parent_->Take(this);
  block->insert(position, op);
}

std::string Operation::name() const {
  auto p_name = info_.name();
  return p_name ? p_name : "";
}

void Operation::Erase() {
  if (auto *parent = GetParent())
    parent->erase(*this);
  else
    Destroy();
}

bool Operation::use_empty() {
  auto res = results();
  return std::all_of(
      res.begin(), res.end(), [](Value result) { return result.use_empty(); });
}

void Operation::ReplaceAllUsesWith(const std::vector<Value> &values) {
  PADDLE_ENFORCE_EQ(
      num_results_,
      values.size(),
      common::errors::InvalidArgument("the num of result should be the same."));
  for (uint32_t i = 0; i < num_results_; ++i) {
    result(i).ReplaceAllUsesWith(values[i]);
  }
}

void Operation::Verify() {
  if (info_) {
    info_.Verify(this);
  }
}

int32_t Operation::ComputeOpResultOffset(uint32_t index) const {
  PADDLE_ENFORCE_LT(
      index,
      num_results_,
      common::errors::PreconditionNotMet(
          "The op result index [%u] must less than results size[%u].",
          index,
          num_results_));
  if (index < OUTLINE_RESULT_IDX) {
    return -static_cast<int32_t>((index + 1u) * sizeof(OpInlineResultImpl));
  }
  constexpr uint32_t anchor = OUTLINE_RESULT_IDX * sizeof(OpInlineResultImpl);
  index = index - MAX_INLINE_RESULT_IDX;
  return -static_cast<int32_t>(index * sizeof(OpOutlineResultImpl) + anchor);
}

int32_t Operation::ComputeOpOperandOffset(uint32_t index) const {
  PADDLE_ENFORCE_LT(
      index,
      num_operands_,
      common::errors::PreconditionNotMet(
          "The op operand index [%u] must less than operands size[%u].",
          index,
          num_operands_));
  return static_cast<int32_t>(index * sizeof(OpOperandImpl) +
                              sizeof(Operation));
}

void Operation::set_value_property(const std::string &key,
                                   const Property &value,
                                   size_t index) {
  if (value_properties_.size() < index + 1) {
    value_properties_.resize(index + 1);
  }
  auto &property_map = value_properties_[index];
  if (property_map.count(key)) {
    property_map[key].second(property_map[key].first);
  }
  property_map[key] = value;
}

void *Operation::value_property(const std::string &key, size_t index) const {
  if (value_properties_.size() < (index + 1)) {
    return nullptr;
  }
  auto &property_map = value_properties_[index];
  auto iter = property_map.find(key);
  return iter == property_map.end() ? nullptr : iter->second.first;
}

#define COMPONENT_IMPL(component_lower, component_upper)                   \
  component_upper##Impl *Operation::component_lower##_impl(uint32_t index) \
      const {                                                              \
    int32_t offset = Compute##component_upper##Offset(index);              \
    return reinterpret_cast<component_upper##Impl *>(                      \
        reinterpret_cast<char *>(const_cast<Operation *>(this)) + offset); \
  }

COMPONENT_IMPL(op_result, OpResult)
COMPONENT_IMPL(op_operand, OpOperand)

IR_API std::ostream &operator<<(std::ostream &os, const Operation &op) {
  op.Print(os);
  return os;
}

}  // namespace pir
