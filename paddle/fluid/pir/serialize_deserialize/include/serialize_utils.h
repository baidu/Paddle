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

#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/serialize_deserialize/include/schema.h"
#include "paddle/fluid/pir/serialize_deserialize/include/third_party.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

namespace pir {
#define COMPRESS_DIALECT_NAME(attr_template)           \
  pir::DialectIdMap::Instance()->GetCompressDialectId( \
      (attr_template).dialect().name())

/**
 * If you need to support serialize type or attr in a new dialect, please add
 * the corresponding method according to the naming convention in the following
 * class, and add a branch of the newly added serialization structure
 * in the implementation function of the method.
 */
class AttrTypeWriter {
 public:
  static Json WriteBuiltInAttr(const pir::Attribute& attr);

  static Json WriteBuiltInType(const pir::Type& type);

  static Json WritePaddleOperatorAttr(const pir::Attribute& attr);

  static Json WritePaddleOperatorType(const pir::Type& type);

  static Json WritePaddleDistType(const pir::Type& type);

  static Json WritePaddleDistAttr(const pir::Attribute& attr);

  static Json WriteControlFlowType(const pir::Type& type);
};
/** serializeTypeToJson is a template function to serialize
 * a pir type to a json object. a pir type may have value or no value
 * Value free types only have ID, while value based types have
 * DATA in addition to ID.
 *
 * If a new pir type is added, which needs to be serialized,
 * it must have a name() method, returning a string which
 * should be different from other types' names.
 * (The name template is t_dialectname_typename).
 * Note: The prefixes t are assumed to represent 'type'.
 *
 * If the pir type has value, it should have a data() method,
 * which returns the value of type. The data() method is better
 * suited to return TYPE  which supported by json like std::vector,
 * std::string, int, float and so on. if not, serializeTypeToJson
 * need to be specialized.
 */

template <typename T>
Json serializeTypeToJson(const T& type) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(type) + "." + type.name();
  return json_obj;
}

/** serializeAttrToJson is a template function to serialize
 * pir attribute to json object. pir attribute usually have
 * value, so it's json object has DATA and ID.
 *
 * If a new pir attr is added, which needs to be serialized,
 * it must have a name() method, returning a string which
 * should be different from other types' names.
 * (The name template is a_dialectname_typename).
 * Note: The prefixes a are assumed to represent 'attribute'.
 *
 * It also need have a data() method, which returns the value of
 * attribute. The data() method is better suited to return TYPE
 * which supported by json like std::vector, std::string, int,
 * float and so on. if not, serializeAttrToJson
 * need to be specialized.
 */

template <typename T>
Json serializeAttrToJson(const T& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();
  json_obj[DATA] = attr.data();
  return json_obj;
}

template <>
Json serializeAttrToJson<pir::FloatAttribute>(const pir::FloatAttribute& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();
  auto data = attr.data();

  if (std::isnan(data)) {
    json_obj[VOID_DATA] = "NaN";
  } else if (std::isinf(data)) {
    if (static_cast<float>(data) > 0.0) {
      json_obj[VOID_DATA] = "INF";
    } else {
      json_obj[VOID_DATA] = "-INF";
    }
  } else {
    json_obj[DATA] = data;
  }
  return json_obj;
}

template <>
Json serializeAttrToJson<pir::DoubleAttribute>(
    const pir::DoubleAttribute& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();
  auto data = attr.data();

  if (std::isnan(data)) {
    json_obj[VOID_DATA] = "NaN";
  } else if (std::isinf(data)) {
    if (static_cast<double>(data) > 0.0) {
      json_obj[VOID_DATA] = "INF";
    } else if (static_cast<double>(data) < 0.0) {
      json_obj[VOID_DATA] = "-INF";
    }
  } else {
    json_obj[DATA] = data;
  }
  return json_obj;
}

#define SERIALIZE_ATTR_TO_JSON(type, data)                          \
  template <>                                                       \
  Json serializeAttrToJson<type>(const type& attr) {                \
    Json json_obj;                                                  \
    json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name(); \
    json_obj[DATA] = data;                                          \
    return json_obj;                                                \
  }

SERIALIZE_ATTR_TO_JSON(pir::StrAttribute, attr.AsString());

SERIALIZE_ATTR_TO_JSON(pir::Complex64Attribute,
                       std::vector({attr.data().real, attr.data().imag}));
SERIALIZE_ATTR_TO_JSON(pir::Complex128Attribute,
                       std::vector({attr.data().real, attr.data().imag}));
SERIALIZE_ATTR_TO_JSON(paddle::dialect::IntArrayAttribute,
                       attr.data().GetData());
SERIALIZE_ATTR_TO_JSON(paddle::dialect::DataTypeAttribute,
                       phi::DataTypeToString(attr.data()));
SERIALIZE_ATTR_TO_JSON(paddle::dialect::DataLayoutAttribute,
                       common::DataLayoutToString(attr.data()));
template <>
Json serializeAttrToJson<paddle::dialect::ScalarAttribute>(
    const paddle::dialect::ScalarAttribute& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();

  Json content = Json::array();
  auto scalar = attr.data();
  auto dtype_ = scalar.dtype();
  content.push_back(DataTypeToString(dtype_));

  if (dtype_ == phi::DataType::FLOAT32) {
    content.push_back(scalar.to<float>());
  } else if (dtype_ == phi::DataType::INT32) {
    content.push_back(scalar.to<int32_t>());
  } else if (dtype_ == phi::DataType::FLOAT64) {
    content.push_back(scalar.to<double>());
  } else if (dtype_ == phi::DataType::INT8) {
    content.push_back(scalar.to<int8_t>());
  } else if (dtype_ == phi::DataType::FLOAT16 ||
             dtype_ == phi::DataType::UINT16 ||
             dtype_ == phi::DataType::BFLOAT16) {
    content.push_back(scalar.to<uint16_t>());
  } else if (dtype_ == phi::DataType::INT16) {
    content.push_back(scalar.to<int16_t>());
  } else if (dtype_ == phi::DataType::INT64) {
    content.push_back(scalar.to<int64_t>());
  } else if (dtype_ == phi::DataType::UINT8) {
    content.push_back(scalar.to<uint8_t>());
  } else if (dtype_ == phi::DataType::UINT32) {
    content.push_back(scalar.to<uint32_t>());
  } else if (dtype_ == phi::DataType::UINT64) {
    content.push_back(scalar.to<uint64_t>());
  } else if (dtype_ == phi::DataType::BOOL) {
    content.push_back(scalar.to<bool>());
  } else if (dtype_ == phi::DataType::COMPLEX64) {
    content.push_back(scalar.to<phi::dtype::complex<float>>().real);
    content.push_back(scalar.to<phi::dtype::complex<float>>().imag);
  } else if (dtype_ == phi::DataType::COMPLEX128) {
    content.push_back(scalar.to<phi::dtype::complex<double>>().real);
    content.push_back(scalar.to<phi::dtype::complex<double>>().imag);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Invalid tensor data type `", dtype_, "`."));
  }
  json_obj[DATA] = content;
  return json_obj;
}

template <>
Json serializeAttrToJson<paddle::dialect::PlaceAttribute>(
    const paddle::dialect::PlaceAttribute& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();
  Json content = Json::array();
  auto place = attr.data();
  content.push_back(static_cast<int8_t>(place.GetType()));
  content.push_back(place.GetDeviceId());    // int8_t
  content.push_back(place.GetDeviceType());  // string
  json_obj[DATA] = content;
  return json_obj;
}

Json writeType(const pir::Type& type) {
  Json type_json = Json::object();
  if (!type) {
    type_json[ID] = NULL_TYPE;
    return type_json;
  }
  if (type.dialect().name() == pir::BuiltinDialect::name()) {
    VLOG(6) << "write BuiltinType ... ";
    return AttrTypeWriter::WriteBuiltInType(type);
  } else if (type.dialect().name() ==
             paddle::dialect::OperatorDialect::name()) {
    VLOG(6) << "write PaddleOperatorType ... ";
    return AttrTypeWriter::WritePaddleOperatorType(type);
  } else if (type.dialect().name() == paddle::dialect::DistDialect::name()) {
    VLOG(6) << "write PaddleDistType ... ";
    return AttrTypeWriter::WritePaddleDistType(type);
  } else if (type.dialect().name() == pir::ControlFlowDialect::name()) {
    VLOG(6) << "write ControlFlowDialect ... ";
    return AttrTypeWriter::WriteControlFlowType(type);
  } else {
    PADDLE_ENFORCE(
        false,
        common::errors::InvalidArgument("Unknown Type %s when write type"));
  }
  VLOG(8) << "Finish write Type ... ";

  return type_json;
}

SERIALIZE_ATTR_TO_JSON(pir::TypeAttribute, writeType(attr.data()));

Json writeAttr(const pir::Attribute& attr) {
  if (!attr) {
    Json attr_json = Json::object();
    attr_json[ID] = NULL_TYPE;
    return attr_json;
  }
  if (attr.dialect().name() == pir::BuiltinDialect::name()) {
    VLOG(8) << "write BuiltinAttr ... ";
    return AttrTypeWriter::WriteBuiltInAttr(attr);
  } else if (attr.dialect().name() ==
             paddle::dialect::OperatorDialect::name()) {
    VLOG(8) << "write PaddleOperatorAttr ... ";
    return AttrTypeWriter::WritePaddleOperatorAttr(attr);
  } else if (attr.dialect().name() == paddle::dialect::DistDialect::name()) {
    VLOG(8) << "write PaddleDistAttr ... ";
    return AttrTypeWriter::WritePaddleDistAttr(attr);
  } else {
    PADDLE_ENFORCE(
        false,
        common::errors::InvalidArgument("Unknown Attr %s when write attr"));
  }

  VLOG(8) << "Finish write attr ... ";

  return Json::object();
}

// ProcessMesh includes: std::vector<int64_t>& shape, std::vector<int64_t>&
// process_ids, std::vector<std::string>& dim_names
template <>
Json serializeAttrToJson<paddle::dialect::ProcessMeshAttribute>(
    const paddle::dialect::ProcessMeshAttribute& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();
  Json content = Json::array();

  content.push_back(attr.shape());
  content.push_back(attr.process_ids());
  content.push_back(attr.dim_names());

  json_obj[DATA] = content;
  return json_obj;
}

// TensorDistAttribute includes: ProcessMeshAttribute mesh_attr,
// std::vector<int64_t> dims_mapping, flat_hash_map<int64_t, phi::ReduceType>
// partial_status;
template <>
Json serializeAttrToJson<paddle::dialect::TensorDistAttribute>(
    const paddle::dialect::TensorDistAttribute& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();
  Json content = Json::array();

  content.push_back(serializeAttrToJson<paddle::dialect::ProcessMeshAttribute>(
      attr.process_mesh_attr()));
  content.push_back(attr.dims_mapping());

  Json map_json = Json::array();
  for (const auto& [key, value] : attr.partial_status()) {
    map_json.push_back(
        std::vector<int64_t>({key, static_cast<int64_t>(value)}));
  }
  content.push_back(map_json);

  json_obj[DATA] = content;
  return json_obj;
}

// OperationDistAttribute includes: ProcessMeshAttribute mesh_attr,
// std::vector<pir::Attribute> operands, std::vector<pir::Attribute> results;
template <>
Json serializeAttrToJson<paddle::dialect::OperationDistAttribute>(
    const paddle::dialect::OperationDistAttribute& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();
  Json content = Json::array();

  content.push_back(serializeAttrToJson<paddle::dialect::ProcessMeshAttribute>(
      attr.process_mesh_attr()));

  Json operands_json = Json::array();
  for (size_t i = 0; i < attr.operands().size(); i++) {
    operands_json.push_back(writeAttr(attr.operands().at(i)));
  }
  content.push_back(operands_json);

  Json results_json = Json::array();
  for (size_t i = 0; i < attr.results().size(); i++) {
    results_json.push_back(writeAttr(attr.results().at(i)));
  }
  content.push_back(results_json);
  content.push_back(attr.chunk_id());

  json_obj[DATA] = content;
  return json_obj;
}

Json AttrTypeWriter::WriteBuiltInAttr(const pir::Attribute& attr) {
  Json attr_json = Json::object();
  if (attr.isa<pir::BoolAttribute>()) {
    VLOG(8) << "write BoolAttribute .";
    return pir::serializeAttrToJson<pir::BoolAttribute>(
        attr.dyn_cast<pir::BoolAttribute>());
  } else if (attr.isa<pir::FloatAttribute>()) {
    VLOG(8) << "write FloatAttribute .";
    return pir::serializeAttrToJson<pir::FloatAttribute>(
        attr.dyn_cast<pir::FloatAttribute>());
  } else if (attr.isa<pir::DoubleAttribute>()) {
    VLOG(8) << "write DoubleAttribute .";
    return pir::serializeAttrToJson<pir::DoubleAttribute>(
        attr.dyn_cast<pir::DoubleAttribute>());
  } else if (attr.isa<pir::Int32Attribute>()) {
    VLOG(8) << "write Int32Attribute .";
    return pir::serializeAttrToJson<pir::Int32Attribute>(
        attr.dyn_cast<pir::Int32Attribute>());
  } else if (attr.isa<pir::Int64Attribute>()) {
    VLOG(8) << "write Int64Attribute .";
    return pir::serializeAttrToJson<pir::Int64Attribute>(
        attr.dyn_cast<pir::Int64Attribute>());
  } else if (attr.isa<pir::IndexAttribute>()) {
    VLOG(8) << "write IndexAttribute .";
    return pir::serializeAttrToJson<pir::IndexAttribute>(
        attr.dyn_cast<pir::IndexAttribute>());
  } else if (attr.isa<pir::ArrayAttribute>()) {
    VLOG(8) << "write ArrayAttribute .";
    auto attr_ = attr.dyn_cast<pir::ArrayAttribute>();
    Json val = Json::array();
    for (size_t i = 0; i < attr_.size(); i++) {
      val.push_back(writeAttr(attr_.at(i)));
    }
    attr_json[ID] = COMPRESS_DIALECT_NAME(attr_) + "." + attr_.name();
    attr_json[DATA] = val;
    return attr_json;
  } else if (attr.isa<pir::TypeAttribute>()) {
    VLOG(8) << "write TypeAttribute .";
    return pir::serializeAttrToJson<pir::TypeAttribute>(
        attr.dyn_cast<pir::TypeAttribute>());
  } else if (attr.isa<pir::TensorNameAttribute>()) {
    VLOG(8) << "write TensorNameAttribute .";
    return pir::serializeAttrToJson<pir::TensorNameAttribute>(
        attr.dyn_cast<pir::TensorNameAttribute>());
  } else if (attr.isa<pir::Complex64Attribute>()) {
    VLOG(8) << "write Complex64Attribute .";
    return pir::serializeAttrToJson<pir::Complex64Attribute>(
        attr.dyn_cast<pir::Complex64Attribute>());
  } else if (attr.isa<pir::Complex128Attribute>()) {
    VLOG(8) << "write Complex128Attribute .";
    return pir::serializeAttrToJson<pir::Complex128Attribute>(
        attr.dyn_cast<pir::Complex128Attribute>());
  } else if (attr.isa<pir::StrAttribute>()) {
    VLOG(8) << "write StrAttribute .";
    return pir::serializeAttrToJson<pir::StrAttribute>(
        attr.dyn_cast<pir::StrAttribute>());
  } else {
    PADDLE_ENFORCE(false,
                   common::errors::InvalidArgument(
                       "Unknown Attr %s when write Builtin dialect attr"));
  }
  return attr_json;
}

template <typename T>
Json serializeTypeToJsonIncludeWriteType(const T& type) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(type) + "." + type.name();
  Json content = Json::array();
  content.push_back(writeType(type.dtype()));

  std::vector<int64_t> dims_;
  for (auto i = 0; i < type.dims().size(); i++) {
    dims_.push_back(type.dims().at(i));
  }
  content.push_back(dims_);

  content.push_back(DataLayoutToString(type.data_layout()));

  content.push_back(type.lod());

  content.push_back(type.offset());
  json_obj[DATA] = content;
  return json_obj;
}
template <>
Json serializeTypeToJsonIncludeWriteType<>(const pir::VectorType& type) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(type) + "." + type.name();
  Json content = Json::array();
  for (auto type_x : type.data()) {
    content.push_back(writeType(type_x));
  }
  json_obj[DATA] = content;
  return json_obj;
}

template <>
Json serializeTypeToJsonIncludeWriteType<paddle::dialect::SparseCooTensorType>(
    const paddle::dialect::SparseCooTensorType& type) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(type) + "." + type.name();
  Json content = Json::array();
  content.push_back(writeType(type.dtype()));

  std::vector<int64_t> dims_;
  for (auto i = 0; i < type.dims().size(); i++) {
    dims_.push_back(type.dims().at(i));
  }
  content.push_back(dims_);

  std::vector<int64_t> non_zero_dims_;
  for (auto i = 0; i < type.non_zero_dims().size(); i++) {
    non_zero_dims_.push_back(type.non_zero_dims().at(i));
  }
  content.push_back(non_zero_dims_);

  content.push_back(DataLayoutToString(type.data_layout()));

  content.push_back(serializeTypeToJsonIncludeWriteType<pir::DenseTensorType>(
      type.non_zero_indices()));

  content.push_back(serializeTypeToJsonIncludeWriteType<pir::DenseTensorType>(
      type.non_zero_elements()));
  json_obj[DATA] = content;
  return json_obj;
}

template <>
Json serializeTypeToJsonIncludeWriteType<paddle::dialect::SparseCsrTensorType>(
    const paddle::dialect::SparseCsrTensorType& type) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(type) + "." + type.name();
  Json content = Json::array();
  content.push_back(writeType(type.dtype()));

  std::vector<int64_t> dims_;
  for (auto i = 0; i < type.dims().size(); i++) {
    dims_.push_back(type.dims().at(i));
  }
  content.push_back(dims_);

  content.push_back(DataLayoutToString(type.data_layout()));

  content.push_back(serializeTypeToJsonIncludeWriteType<pir::DenseTensorType>(
      type.non_zero_crows()));
  content.push_back(serializeTypeToJsonIncludeWriteType<pir::DenseTensorType>(
      type.non_zero_cols()));
  content.push_back(serializeTypeToJsonIncludeWriteType<pir::DenseTensorType>(
      type.non_zero_elements()));
  json_obj[DATA] = content;
  return json_obj;
}

template <>
Json serializeTypeToJsonIncludeWriteType<paddle::dialect::DenseTensorArrayType>(
    const paddle::dialect::DenseTensorArrayType& type) {
  Json json_obj = Json::object();
  json_obj[ID] = COMPRESS_DIALECT_NAME(type) + "." + type.name();
  Json content = Json::array();
  content.push_back(writeType(type.dtype()));

  std::vector<int64_t> dims_;
  for (auto i = 0; i < type.dims().size(); i++) {
    dims_.push_back(type.dims().at(i));
  }
  content.push_back(dims_);

  content.push_back(DataLayoutToString(type.data_layout()));

  json_obj[DATA] = content;
  return json_obj;
}

template <>
Json serializeTypeToJsonIncludeWriteType<paddle::dialect::DistDenseTensorType>(
    const paddle::dialect::DistDenseTensorType& type) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(type) + "." + type.name();
  Json content = Json::array();

  // serialize pir::DenseTensorType dense_tensor_type;
  content.push_back(serializeTypeToJsonIncludeWriteType<pir::DenseTensorType>(
      type.dense_tensor_type()));

  // serialize TensorDistAttribute tensor_dist_attr;
  content.push_back(serializeAttrToJson<paddle::dialect::TensorDistAttribute>(
      type.tensor_dist_attr()));

  // serialize common::DDim local_ddim;
  std::vector<int64_t> local_ddim_;
  for (auto i = 0; i < type.local_ddim().size(); i++) {
    local_ddim_.push_back(type.local_ddim().at(i));
  }
  content.push_back(local_ddim_);

  json_obj[DATA] = content;
  return json_obj;
}

Json AttrTypeWriter::WriteBuiltInType(const pir::Type& type) {
  Json type_json = Json::object();
  if (type.isa<pir::BoolType>()) {
    VLOG(8) << "Write BoolType ... ";
    return pir::serializeTypeToJson<pir::BoolType>(
        type.dyn_cast<pir::BoolType>());
  } else if (type.isa<pir::BFloat16Type>()) {
    VLOG(8) << "Write BFloat16Type ... ";
    return pir::serializeTypeToJson<pir::BFloat16Type>(
        type.dyn_cast<pir::BFloat16Type>());
  } else if (type.isa<pir::Float16Type>()) {
    VLOG(8) << "Write Float16Type ... ";
    return pir::serializeTypeToJson<pir::Float16Type>(
        type.dyn_cast<pir::Float16Type>());
  } else if (type.isa<pir::Float32Type>()) {
    VLOG(8) << "Write Float32Type ... ";
    return pir::serializeTypeToJson<pir::Float32Type>(
        type.dyn_cast<pir::Float32Type>());
  } else if (type.isa<pir::Float64Type>()) {
    VLOG(8) << "Write Float64Type ... ";
    return pir::serializeTypeToJson<pir::Float64Type>(
        type.dyn_cast<pir::Float64Type>());
  } else if (type.isa<pir::Int8Type>()) {
    VLOG(8) << "Write Int8Type ... ";
    return pir::serializeTypeToJson<pir::Int8Type>(
        type.dyn_cast<pir::Int8Type>());
  } else if (type.isa<pir::UInt8Type>()) {
    VLOG(8) << "Write UInt8Type ... ";
    return pir::serializeTypeToJson<pir::UInt8Type>(
        type.dyn_cast<pir::UInt8Type>());
  } else if (type.isa<pir::Int16Type>()) {
    VLOG(8) << "Write Int16Type ... ";
    return pir::serializeTypeToJson<pir::Int16Type>(
        type.dyn_cast<pir::Int16Type>());
  } else if (type.isa<pir::Int32Type>()) {
    VLOG(8) << "Write Int32Type ... ";
    return pir::serializeTypeToJson<pir::Int32Type>(
        type.dyn_cast<pir::Int32Type>());
  } else if (type.isa<pir::Int64Type>()) {
    VLOG(8) << "Write Int64Type ... ";
    return pir::serializeTypeToJson<pir::Int64Type>(
        type.dyn_cast<pir::Int64Type>());
  } else if (type.isa<pir::IndexType>()) {
    VLOG(8) << "Write IndexType ... ";
    return pir::serializeTypeToJson<pir::IndexType>(
        type.dyn_cast<pir::IndexType>());
  } else if (type.isa<pir::Complex64Type>()) {
    VLOG(8) << "Write Complex64Type ... ";
    return pir::serializeTypeToJson<pir::Complex64Type>(
        type.dyn_cast<pir::Complex64Type>());
  } else if (type.isa<pir::Complex128Type>()) {
    VLOG(8) << "Write Complex128Type ... ";
    return pir::serializeTypeToJson<pir::Complex128Type>(
        type.dyn_cast<pir::Complex128Type>());
    // NOTE(Ruting) those Types need call writeType which make build error
    //  when use template func serializeTypeToJson
  } else if (type.isa<pir::VectorType>()) {
    VLOG(8) << "Write VectorType ... ";
    return pir::serializeTypeToJsonIncludeWriteType<pir::VectorType>(
        type.dyn_cast<pir::VectorType>());
  } else if (type.isa<pir::DenseTensorType>()) {
    VLOG(8) << "Write DenseTensorType ... ";
    return pir::serializeTypeToJsonIncludeWriteType<pir::DenseTensorType>(
        type.dyn_cast<pir::DenseTensorType>());
  } else if (type.isa<pir::UndefinedType>()) {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Unexpected type pir::UndefinedType, "
        "it should be replace with a concrete type when ArrayWrite."));
  } else {
    PADDLE_ENFORCE(false,
                   common::errors::InvalidArgument(
                       "Unknown Type when write builtin dialect type"));
  }
  return type_json;
}

Json AttrTypeWriter::WritePaddleOperatorAttr(const pir::Attribute& attr) {
  if (attr.isa<paddle::dialect::IntArrayAttribute>()) {
    VLOG(8) << "write IntArrayAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::IntArrayAttribute>(
        attr.dyn_cast<paddle::dialect::IntArrayAttribute>());
  } else if (attr.isa<paddle::dialect::ScalarAttribute>()) {
    VLOG(8) << "write ScalarAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::ScalarAttribute>(
        attr.dyn_cast<paddle::dialect::ScalarAttribute>());
  } else if (attr.isa<paddle::dialect::DataTypeAttribute>()) {
    VLOG(8) << "write DataTypeAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::DataTypeAttribute>(
        attr.dyn_cast<paddle::dialect::DataTypeAttribute>());
  } else if (attr.isa<paddle::dialect::PlaceAttribute>()) {
    VLOG(8) << "write PlaceAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::PlaceAttribute>(
        attr.dyn_cast<paddle::dialect::PlaceAttribute>());
  } else if (attr.isa<paddle::dialect::DataLayoutAttribute>()) {
    VLOG(8) << "write DataLayoutAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::DataLayoutAttribute>(
        attr.dyn_cast<paddle::dialect::DataLayoutAttribute>());
  } else {
    PADDLE_ENFORCE(
        false,
        common::errors::InvalidArgument(
            "Unknown Attr %s when write paddle.operatordialect attr"));
  }
  return Json::object();
}

Json AttrTypeWriter::WritePaddleOperatorType(const pir::Type& type) {
  Json type_json = Json::object();
  if (type.isa<paddle::dialect::DenseTensorArrayType>()) {
    VLOG(8) << "Write DenseTensorArrayType ... ";
    return pir::serializeTypeToJsonIncludeWriteType<
        paddle::dialect::DenseTensorArrayType>(
        type.dyn_cast<paddle::dialect::DenseTensorArrayType>());
  } else if (type.isa<paddle::dialect::SelectedRowsType>()) {
    VLOG(8) << "Write SelectedRowsType ... ";
    return pir::serializeTypeToJsonIncludeWriteType<
        paddle::dialect::SelectedRowsType>(
        type.dyn_cast<paddle::dialect::SelectedRowsType>());
  } else if (type.isa<paddle::dialect::SparseCooTensorType>()) {
    VLOG(8) << "Write SparseCooTensorType ... ";
    return pir::serializeTypeToJsonIncludeWriteType<
        paddle::dialect::SparseCooTensorType>(
        type.dyn_cast<paddle::dialect::SparseCooTensorType>());
  } else if (type.isa<paddle::dialect::SparseCsrTensorType>()) {
    VLOG(8) << "Write SparseCsrTensorType ... ";
    return pir::serializeTypeToJsonIncludeWriteType<
        paddle::dialect::SparseCsrTensorType>(
        type.dyn_cast<paddle::dialect::SparseCsrTensorType>());
  } else {
    PADDLE_ENFORCE(false,
                   common::errors::InvalidArgument(
                       "Unknown Type when write paddle.operatordialect type"));
    return Json::object();
  }
}

Json AttrTypeWriter::WritePaddleDistType(const pir::Type& type) {
  Json type_json = Json::object();
  if (type.isa<paddle::dialect::DistDenseTensorType>()) {
    VLOG(8) << "Write DistDenseTensorType ... ";
    return pir::serializeTypeToJsonIncludeWriteType<
        paddle::dialect::DistDenseTensorType>(
        type.dyn_cast<paddle::dialect::DistDenseTensorType>());
  } else {
    PADDLE_ENFORCE(false,
                   common::errors::InvalidArgument(
                       "Unknown Type when write paddle.dist_dialect type"));
    return Json::object();
  }
}

Json AttrTypeWriter::WritePaddleDistAttr(const pir::Attribute& attr) {
  if (attr.isa<paddle::dialect::ProcessMeshAttribute>()) {
    VLOG(8) << "write ProcessMeshAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::ProcessMeshAttribute>(
        attr.dyn_cast<paddle::dialect::ProcessMeshAttribute>());
  } else if (attr.isa<paddle::dialect::TensorDistAttribute>()) {
    VLOG(8) << "write TensorDistAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::TensorDistAttribute>(
        attr.dyn_cast<paddle::dialect::TensorDistAttribute>());
  } else if (attr.isa<paddle::dialect::OperationDistAttribute>()) {
    VLOG(8) << "write OperationDistAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::OperationDistAttribute>(
        attr.dyn_cast<paddle::dialect::OperationDistAttribute>());
  } else {
    PADDLE_ENFORCE(
        false,
        common::errors::InvalidArgument(
            "Unknown Attr %s when write paddle.operatordialect attr"));
  }
  return Json::object();
}

Json AttrTypeWriter::WriteControlFlowType(const pir::Type& type) {
  Json type_json = Json::object();
  if (type.isa<pir::StackType>()) {
    VLOG(8) << "Write StackType ... ";
    return pir::serializeTypeToJson<pir::StackType>(
        type.dyn_cast<pir::StackType>());
  } else if (type.isa<pir::InletType>()) {
    VLOG(8) << "Write InletType ... ";
    return pir::serializeTypeToJson<pir::InletType>(
        type.dyn_cast<pir::InletType>());
  } else if (type.isa<pir::OutletType>()) {
    VLOG(8) << "Write OutletType ... ";
    return pir::serializeTypeToJson<pir::OutletType>(
        type.dyn_cast<pir::OutletType>());
  } else {
    PADDLE_ENFORCE(false,
                   common::errors::InvalidArgument(
                       "Unknown Type when write controlflow dialect type"));
  }
  return type_json;
}

}  // namespace pir
