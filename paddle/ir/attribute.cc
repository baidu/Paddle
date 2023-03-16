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

#include "paddle/ir/attribute.h"
#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/dialect.h"

namespace ir {
IrContext *Attribute::ir_context() const { return dialect().ir_context(); }

NamedAttribute::NamedAttribute(StrAttribute name, Attribute value)
    : name_(name), value_(value) {}

StrAttribute NamedAttribute::name() const {
  return name_.dyn_cast<StrAttribute>();
}

Attribute NamedAttribute::value() const { return value_; }

void NamedAttribute::SetName(StrAttribute name) { name_ = name; }

void NamedAttribute::SetValue(Attribute value) { value_ = value; }

bool NamedAttribute::operator<(const NamedAttribute &right) const {
  return name().compare(right.name()) < 0;
}

bool NamedAttribute::operator==(const NamedAttribute &right) const {
  return name() == right.name() && value() == right.value();
}

bool NamedAttribute::operator!=(const NamedAttribute &right) const {
  return !(*this == right);
}
}  // namespace ir
