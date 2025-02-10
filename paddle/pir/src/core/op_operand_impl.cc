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

#include "paddle/pir/src/core/op_operand_impl.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/src/core/value_impl.h"

namespace pir::detail {

pir::Operation *OpOperandImpl::owner() const { return owner_; }

pir::detail::OpOperandImpl *OpOperandImpl::next_use() { return next_use_; }

pir::Value OpOperandImpl::source() const { return source_; }

void OpOperandImpl::set_source(Value source) {
  RemoveFromUdChain();
  if (!source) {
    return;
  }
  source_ = source;
  InsertToUdChain();
}

OpOperandImpl::OpOperandImpl(pir::Value source, pir::Operation *owner)
    : source_(source), owner_(owner) {
  if (!source) {
    return;
  }
  InsertToUdChain();
}

uint32_t OpOperandImpl::index() const {
  const char *start =
      reinterpret_cast<const char *>(owner_) + sizeof(Operation);
  const char *end = reinterpret_cast<const char *>(this);
  return (end - start) / sizeof(OpOperandImpl);
}

void OpOperandImpl::InsertToUdChain() {
  prev_use_addr_ = source_.impl()->first_use_addr();
  next_use_ = source_.impl()->first_use();
  if (next_use_) {
    next_use_->prev_use_addr_ = &next_use_;
  }
  source_.impl()->set_first_use(this);
}

void OpOperandImpl::RemoveFromUdChain() {
  if (!source_) return;
  if (!prev_use_addr_) return;
  if (prev_use_addr_ == source_.impl()->first_use_addr()) {
    /// NOTE: In ValueImpl, first_use_offsetted_by_index_ use lower three bits
    /// storage index information, so need to be updated using the set_first_use
    /// method here.
    source_.impl()->set_first_use(next_use_);
  } else {
    *prev_use_addr_ = next_use_;
  }
  if (next_use_) {
    next_use_->prev_use_addr_ = prev_use_addr_;
  }
  next_use_ = nullptr;
  prev_use_addr_ = nullptr;
  source_ = nullptr;
}

OpOperandImpl::~OpOperandImpl() { RemoveFromUdChain(); }

}  // namespace pir::detail
