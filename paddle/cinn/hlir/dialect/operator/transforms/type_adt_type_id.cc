// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/type_adt_type_id.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"

namespace cinn::dialect::ir {

TypeAdtTypeId GetTypeAdtTypeId(const pir::Type& type) {
  if (!type) {
    return ::common::AdtTypeId<NullType>{};
  }
#define RETURN_TYPE_TYPE_ID_IF_MATCH(cls) \
  if (type.isa<cls>()) return ::common::AdtTypeId<cls>{};
  FOR_EACH_PIR_ALTERNATIVE_TYPE(RETURN_TYPE_TYPE_ID_IF_MATCH)
#undef RETURN_TYPE_TYPE_ID_IF_MATCH
  return ::common::AdtTypeId<UnclassifiedType>{};
}

}  // namespace cinn::dialect::ir
