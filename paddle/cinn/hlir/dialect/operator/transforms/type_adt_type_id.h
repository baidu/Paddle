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

#include "paddle/common/adt_type_id.h"

namespace pir {
class Type;
class VectorType;
class DenseTensorType;
class BFloat16Type;
class Float16Type;
class Float32Type;
class Float64Type;
class Int8Type;
class UInt8Type;
class Int16Type;
class Int32Type;
class Int64Type;
class IndexType;
class BoolType;
class Complex64Type;
class Complex128Type;

}  // namespace pir

namespace paddle::dialect {

class SelectedRowsType;
class DenseTensorArrayType;
class SparseCooTensorType;
class SparseCsrTensorType;

}  // namespace paddle::dialect

// clang-format off
#define FOR_EACH_PIR_ALTERNATIVE_TYPE(__macro)     \
  __macro(::pir::VectorType)                        \
  __macro(::pir::DenseTensorType)                   \
  __macro(::pir::BFloat16Type)                      \
  __macro(::pir::Float16Type)                       \
  __macro(::pir::Float32Type)                       \
  __macro(::pir::Float64Type)                       \
  __macro(::pir::Int8Type)                          \
  __macro(::pir::UInt8Type)                         \
  __macro(::pir::Int16Type)                         \
  __macro(::pir::Int32Type)                         \
  __macro(::pir::Int64Type)                         \
  __macro(::pir::IndexType)                         \
  __macro(::pir::BoolType)                          \
  __macro(::pir::Complex64Type)                     \
  __macro(::pir::Complex128Type)                    \
  __macro(::paddle::dialect::SelectedRowsType)      \
  __macro(::paddle::dialect::DenseTensorArrayType)  \
  __macro(::paddle::dialect::SparseCooTensorType)   \
  __macro(::paddle::dialect::SparseCsrTensorType)
// clang-format on

namespace cinn::dialect::ir {

class NullType;
class UnclassifiedType;

using TypeAdtTypeIdBase =
    ::common::AdtBaseTypeId<NullType,
#define MAKE_TYPE_ADT_TYPE_ID_ALTERNATIVE(cls) cls,
                            FOR_EACH_PIR_ALTERNATIVE_TYPE(
                                MAKE_TYPE_ADT_TYPE_ID_ALTERNATIVE)
#undef MAKE_TYPE_ADT_TYPE_ID_ALTERNATIVE
                                UnclassifiedType>;

struct TypeAdtTypeId : public TypeAdtTypeIdBase {
  using TypeAdtTypeIdBase::TypeAdtTypeIdBase;
};

TypeAdtTypeId GetTypeAdtTypeId(const pir::Type& type);

}  // namespace cinn::dialect::ir
