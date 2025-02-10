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

#include "paddle/common/enforce.h"
#include "paddle/pir/src/core/value_impl.h"

namespace {
uint64_t GenerateId() {
  static std::atomic<std::uint64_t> uid{0};
  return ++uid;
}
}  // namespace

namespace pir::detail {
void ValueImpl::set_first_use(OpOperandImpl *first_use) {
  uint32_t offset = kind();
  uintptr_t ptr = reinterpret_cast<uintptr_t>(first_use) + offset;
  first_use_offsetted_by_kind_ = reinterpret_cast<OpOperandImpl *>(ptr);
  VLOG(10) << "The index of this value is: " << offset
           << ". The address of this value is: " << this
           << ". This value first use is: " << first_use << ".";
}

std::string ValueImpl::PrintUdChain() {
  std::stringstream result;
  result << "Value[" << this << "] -> ";
  OpOperandImpl *tmp = first_use();
  if (tmp) {
    result << "OpOperand[" << reinterpret_cast<void *>(tmp) << "] -> ";
    while (tmp->next_use() != nullptr) {
      result << "OpOperand[" << reinterpret_cast<void *>(tmp->next_use())
             << "] -> ";
      tmp = tmp->next_use();
    }
  }
  result << "nullptr";
  return result.str();
}
ValueImpl::ValueImpl(Type type, uint32_t kind) : id_(GenerateId()) {
  PADDLE_ENFORCE_LE(
      kind,
      BLOCK_ARG_IDX,
      common::errors::PreconditionNotMet(
          "The kind of value_impl[%u] must not bigger than BLOCK_ARG_IDX(7)",
          kind));
  type_ = type;
  uintptr_t ptr = reinterpret_cast<uintptr_t>(nullptr) + kind;
  first_use_offsetted_by_kind_ = reinterpret_cast<OpOperandImpl *>(ptr);
  VLOG(10) << "Construct a ValueImpl whose's kind is " << kind
           << ". The value_impl address is: " << this;
}

}  // namespace pir::detail
