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
#include <sstream>
#include <unordered_set>

#include "paddle/common/errors.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/utils/string/string_helper.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#endif

namespace paddle {
namespace dialect {

const std::unordered_set<std::string> LegacyOpList = {
    DistributedPushSparseOp::name(),
    SendV2Op::name(),
    RecvV2Op::name(),
    CAllreduceSumOp::name(),
    CAllreduceSum_Op::name(),
    PushDenseOp::name(),
    PullBoxSparseOp::name(),
    PushBoxSparseOp::name(),
    PushSparseV2Op::name(),
    SendAndRecvOp::name()};

enum class AttrType {
  UNDEFINED = 0,
  BOOL,
  INT32,
  INT64,

  FLOAT,
  DOUBLE,

  ARRAY,
  INT_ARRAY,

  SCALAR,
  DATA_TYPE,
  DATA_LAYOUT,
  PLACE,

  STRING,

  TENSOR_NAME,

  NUM_ATTR_TYPES,
};

static inline AttrType GetAttributeType(const pir::Attribute& attr) {
  if (attr.isa<pir::BoolAttribute>()) {
    return AttrType::BOOL;
  } else if (attr.isa<pir::FloatAttribute>()) {
    return AttrType::FLOAT;
  } else if (attr.isa<pir::DoubleAttribute>()) {
    return AttrType::DOUBLE;
  } else if (attr.isa<pir::Int32Attribute>()) {
    return AttrType::INT32;
  } else if (attr.isa<pir::Int64Attribute>()) {
    return AttrType::INT64;
  } else if (attr.isa<pir::ArrayAttribute>()) {
    return AttrType::ARRAY;
  } else if (attr.isa<pir::StrAttribute>()) {
    return AttrType::STRING;
  } else if (attr.isa<paddle::dialect::IntArrayAttribute>()) {
    return AttrType::INT_ARRAY;
  } else if (attr.isa<paddle::dialect::DataTypeAttribute>()) {
    return AttrType::DATA_TYPE;
  } else if (attr.isa<paddle::dialect::PlaceAttribute>()) {
    return AttrType::PLACE;
  } else if (attr.isa<pir::TensorNameAttribute>()) {
    return AttrType::TENSOR_NAME;
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported ir Attribute type when casting it into "
        "AttrType."));
  }
}

template <typename T>
static std::function<T(const pir::Attribute& attr)> GetAttrCast(
    AttrType attr_type) {
  std::unordered_map<AttrType, std::function<T(const pir::Attribute& attr)>>
      kAttrCastMap = {
          {AttrType::BOOL,
           [](const pir::Attribute& attr) {
             return T{attr.dyn_cast<pir::BoolAttribute>().data()};
           }},
          {AttrType::FLOAT,
           [](const pir::Attribute& attr) {
             return T{attr.dyn_cast<pir::FloatAttribute>().data()};
           }},
          {AttrType::DOUBLE,
           [](const pir::Attribute& attr) {
             return T{attr.dyn_cast<pir::DoubleAttribute>().data()};
           }},
          {AttrType::INT32,
           [](const pir::Attribute& attr) {
             return T{attr.dyn_cast<pir::Int32Attribute>().data()};
           }},
          {AttrType::INT64,
           [](const pir::Attribute& attr) {
             return T{attr.dyn_cast<pir::Int64Attribute>().data()};
           }},
          {AttrType::INT_ARRAY,
           [](const pir::Attribute& attr) {
             return T{attr.dyn_cast<paddle::dialect::IntArrayAttribute>()
                          .data()
                          .GetData()};
           }},
          {AttrType::STRING,
           [](const pir::Attribute& attr) {
             return T{attr.dyn_cast<pir::StrAttribute>().AsString()};
           }},
          {AttrType::DATA_TYPE,
           [](const pir::Attribute& attr) {
             return T{
                 attr.dyn_cast<paddle::dialect::DataTypeAttribute>().data()};
           }},
          {AttrType::PLACE,
           [](const pir::Attribute& attr) {
             return T{attr.dyn_cast<paddle::dialect::PlaceAttribute>().data()};
           }},
          {AttrType::TENSOR_NAME,
           [](const pir::Attribute& attr) {
             return T{attr.dyn_cast<pir::TensorNameAttribute>().data()};
           }},
          {AttrType::ARRAY,
           [](const pir::Attribute& attr) {
             auto attr_vec = attr.dyn_cast<pir::ArrayAttribute>().AsVector();
             if (attr_vec.empty()) {
               return T{std::vector<int>()};
             }
             AttrType element_type = GetAttributeType(attr_vec[0]);

             if (element_type == AttrType::BOOL) {
               std::vector<bool> vec_bools;
               vec_bools.reserve(attr_vec.size());
               for (auto vec_element : attr_vec) {
                 vec_bools.push_back(
                     vec_element.dyn_cast<pir::BoolAttribute>().data());
               }
               return T{vec_bools};
             } else if (element_type == AttrType::INT32) {
               std::vector<int> vec_int32;
               vec_int32.reserve(attr_vec.size());
               for (auto vec_element : attr_vec) {
                 vec_int32.push_back(
                     vec_element.dyn_cast<pir::Int32Attribute>().data());
               }
               return T{vec_int32};
             } else if (element_type == AttrType::INT64) {
               std::vector<int64_t> vec_int64;
               vec_int64.reserve(attr_vec.size());
               for (auto vec_element : attr_vec) {
                 vec_int64.push_back(
                     vec_element.dyn_cast<pir::Int64Attribute>().data());
               }
               return T{vec_int64};
             } else if (element_type == AttrType::FLOAT) {
               std::vector<float> vec_float;
               vec_float.reserve(attr_vec.size());
               for (auto vec_element : attr_vec) {
                 vec_float.push_back(
                     vec_element.dyn_cast<pir::FloatAttribute>().data());
               }
               return T{vec_float};
             } else if (element_type == AttrType::DOUBLE) {
               std::vector<double> vec_double;
               vec_double.reserve(attr_vec.size());
               for (auto vec_element : attr_vec) {
                 vec_double.push_back(
                     vec_element.dyn_cast<pir::DoubleAttribute>().data());
               }
               return T{vec_double};
             } else if (element_type == AttrType::STRING) {
               std::vector<std::string> vec_string;
               vec_string.reserve(attr_vec.size());
               for (auto vec_element : attr_vec) {
                 vec_string.push_back(
                     vec_element.dyn_cast<pir::StrAttribute>().AsString());
               }
               return T{vec_string};
             } else {
               PADDLE_THROW(common::errors::Unimplemented(
                   "Unsupported ir Attribute type when casting it into "
                   "vector."));
             }
           }},
      };
  return kAttrCastMap[attr_type];
}

VariantType GetAttributeData(const pir::Attribute& attr) {
  AttrType attr_type = GetAttributeType(attr);
  return GetAttrCast<VariantType>(attr_type)(attr);
}

paddle::any TransAttrToAny(const pir::Attribute& attr) {
  AttrType attr_type = GetAttributeType(attr);
  return GetAttrCast<paddle::any>(attr_type)(attr);
}

bool IsLegacyOp(const std::string& name) { return LegacyOpList.count(name); }

bool IsEmptyValue(const pir::Value& value) {
  return !value.impl() || !value.type();
}

std::vector<int64_t> GetInt64Vector(const pir::Attribute& attr) {
  PADDLE_ENFORCE_EQ(attr.isa<pir::ArrayAttribute>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "attribute MUST be a pir::ArrayAttribute"));
  auto attr_vec = attr.dyn_cast<pir::ArrayAttribute>().AsVector();

  std::vector<int64_t> vec_int64;
  for (auto vec_element : attr_vec) {
    PADDLE_ENFORCE_EQ(
        vec_element.isa<pir::Int64Attribute>(),
        true,
        common::errors::PreconditionNotMet("element MUST be a Int64Attribute"));
    vec_int64.push_back(vec_element.dyn_cast<pir::Int64Attribute>().data());
  }

  return vec_int64;
}

std::set<std::string> GetRegisterDataType(const std::string& op_name) {
  std::string non_inplace_op_name;
  if (paddle::string::ends_with(op_name, "_")) {
    non_inplace_op_name = op_name.substr(0, op_name.size() - 1);
  }

  std::set<std::string> data_type;
  auto& phi_kernels = phi::KernelFactory::Instance().kernels();
  for (auto& kernel_pair : phi_kernels) {
    auto fluid_op_name = phi::TransToFluidOpName(kernel_pair.first);
    if (kernel_pair.first != op_name && fluid_op_name != op_name &&
        kernel_pair.first != non_inplace_op_name &&
        fluid_op_name != non_inplace_op_name) {
      continue;
    }
    for (auto& info_pair : kernel_pair.second) {
      data_type.insert(phi::DataTypeToString(info_pair.first.dtype()));
    }
  }
  if (data_type.empty()) {
    VLOG(6) << "No data type is registered for " << op_name;
  }
  return data_type;
}

phi::DataType GetValueDataType(const pir::Type& type) {
  if (type.isa<pir::DenseTensorType>()) {
    return dialect::TransToPhiDataType(
        type.dyn_cast<pir::DenseTensorType>().dtype());
  } else if (type.isa<paddle::dialect::SelectedRowsType>()) {
    return dialect::TransToPhiDataType(
        type.dyn_cast<paddle::dialect::SelectedRowsType>().dtype());
  } else if (type.isa<paddle::dialect::SparseCooTensorType>()) {
    return dialect::TransToPhiDataType(
        type.dyn_cast<paddle::dialect::SparseCooTensorType>().dtype());
  } else if (type.isa<paddle::dialect::SparseCsrTensorType>()) {
    return dialect::TransToPhiDataType(
        type.dyn_cast<paddle::dialect::SparseCsrTensorType>().dtype());
  } else if (type.isa<DenseTensorArrayType>()) {
    return dialect::TransToPhiDataType(
        type.dyn_cast<DenseTensorArrayType>().dtype());
  } else if (type.isa<pir::VectorType>()) {
    auto vec_value = type.dyn_cast<pir::VectorType>();
    if (vec_value.size() > 0) {
      return GetValueDataType(vec_value[0]);
    } else {
      return phi::DataType::UNDEFINED;
    }
  } else {
    PADDLE_THROW(common::errors::InvalidType(
        "Not support op type %s in ConvertOpTypeToKernelType.", type));
    PADDLE_THROW(
        common::errors::InvalidType("Currently, we can only get dtype for "
                                    "DenseTensorType and SelectedRowsType."));
  }
}

phi::DataType GetValueDataType(const pir::Value& value) {
  if (value.impl() == nullptr) {
    return phi::DataType::UNDEFINED;
  }
  return GetValueDataType(value.type());
}

void DoValueCheck(const pir::Value& value,
                  const std::string& input_name,
                  const std::set<std::string>& expected_dtype,
                  const std::string& op_name) {
  std::string value_type = phi::DataTypeToString(GetValueDataType(value));
  if (expected_dtype.find(value_type) == expected_dtype.end()) {
    std::ostringstream joined;
    std::copy(expected_dtype.begin(),
              expected_dtype.end(),
              std::ostream_iterator<std::string>(joined, ", "));
    PADDLE_THROW(common::errors::InvalidType(
        "Check data type error for op: %s, input: %s, %s.dtype: %s, and "
        "expected_dtype: %s",
        op_name,
        input_name,
        input_name,
        value_type,
        joined.str()));
  }
}

void CheckValueDataType(const pir::Value& value,
                        const std::string& input_name,
                        const std::string& op_name) {
  VLOG(6) << "CheckValueDataType for " << op_name << ", input: " << input_name;
  std::set<std::string> expected_dtype = GetRegisterDataType(op_name);
  DoValueCheck(value, input_name, expected_dtype, op_name);
}

bool IsSameDataTypeForValues(const std::vector<pir::Value>& vector_value) {
  if (vector_value.size() <= 1) {
    return true;
  }
  auto dtype = GetValueDataType(vector_value[0]);
  for (size_t i = 1; i < vector_value.size(); ++i) {
    if (GetValueDataType(vector_value[i]) != dtype) {
      return false;
    }
  }
  return true;
}

void CheckVectorOfValueDataType(const std::vector<pir::Value>& vector_value,
                                const std::string& input_name,
                                const std::string& op_name) {
  VLOG(6) << "CheckVectorOfValueDataType for " << op_name
          << ", input: " << input_name;
  if (vector_value.size() == 0) {
    return;
  }
  if (!IsSameDataTypeForValues(vector_value)) {
    PADDLE_THROW(common::errors::InvalidType(
        "All the Values in the input must have the same data type."));
  }
  std::set<std::string> expected_dtype = GetRegisterDataType(op_name);
  DoValueCheck(vector_value[0], input_name, expected_dtype, op_name);
}

void CheckDataType(const phi::DataType& dtype,
                   const std::string& dtype_name,
                   const std::string& op_name) {
  VLOG(6) << "CheckDataType for " << op_name << ", input dtype: " << dtype_name;
  std::set<std::string> expected_dtype = GetRegisterDataType(op_name);

  std::string str_dtype = phi::DataTypeToString(dtype);
  if (expected_dtype.find(str_dtype) == expected_dtype.end()) {
    std::ostringstream joined;
    std::copy(expected_dtype.begin(),
              expected_dtype.end(),
              std::ostream_iterator<std::string>(joined, ", "));
    PADDLE_THROW(common::errors::InvalidType(
        "Check data type error for op: %s, dtype: %s, and "
        "expected_dtype: %s",
        op_name,
        str_dtype,
        joined.str()));
  }
}

void CheckDataTypeOrValue(const phi::DataType& dtype,
                          const std::string& dtype_name,
                          const pir::Value& value,
                          const std::string& value_name,
                          const std::string& op_name) {
  if (dtype == phi::DataType::UNDEFINED) {
    CheckValueDataType(value, value_name, op_name);
  } else {
    CheckDataType(dtype, dtype_name, op_name);
  }
}

std::vector<int64_t> ParseValueShape(const pir::Value& shape,
                                     bool* is_from_tensor) {
  std::vector<int64_t> vec_shape;
  if (shape.isa<pir::OpResult>() &&
      shape.defining_op()->isa<paddle::dialect::FullIntArrayOp>()) {
    vec_shape = paddle::dialect::GetInt64Vector(
        shape.defining_op()
            ->dyn_cast<paddle::dialect::FullIntArrayOp>()
            .attribute("value"));
  } else if (shape.isa<pir::OpResult>() &&
             shape.defining_op()->isa<paddle::dialect::FullOp>()) {
    auto shape_item = shape.defining_op()
                          ->dyn_cast<paddle::dialect::FullOp>()
                          .attribute("value")
                          .dyn_cast<paddle::dialect::ScalarAttribute>()
                          .data()
                          .to<double>();
    vec_shape = {static_cast<int64_t>(shape_item)};
  } else if (shape.isa<pir::OpResult>() &&
             shape.defining_op()->isa<paddle::dialect::StackOp>()) {
    std::vector<pir::Value> inputs =
        shape.defining_op()->operand_source(0).defining_op()->operands_source();
    for (auto item : inputs) {
      auto tmp = ParseValueShape(item, is_from_tensor);
      vec_shape.insert(vec_shape.end(), tmp.begin(), tmp.end());
    }
  } else if (shape.isa<pir::OpResult>() &&
             (shape.defining_op()->isa<paddle::dialect::ShapeOp>() ||
              shape.defining_op()->isa<paddle::dialect::Shape64Op>()) &&
             shape.type().isa<paddle::dialect::DenseTensorType>()) {
    // tensor_shape may come from shape op
    // x0.shape = [-1,3]
    // tensor_shape = shape(x0)
    // y = reshape(x, tensor_shape)
    pir::Value inputs = shape.defining_op()->operand_source(0);
    vec_shape = common::vectorize(
        inputs.type().dyn_cast<paddle::dialect::DenseTensorType>().dims());
    *is_from_tensor = true;
  } else if (shape.isa<pir::OpResult>() &&
             shape.defining_op()->isa<paddle::dialect::ConcatOp>()) {
    // tensor_shape may come from concat
    // tensor_shape = concat([full(1), full(2)])
    // y = reshape(x, tensor_shape)
    std::vector<pir::Value> inputs =
        shape.defining_op()->operand_source(0).defining_op()->operands_source();
    for (auto item : inputs) {
      auto tmp = ParseValueShape(item, is_from_tensor);
      vec_shape.insert(vec_shape.end(), tmp.begin(), tmp.end());
    }
  } else if (shape.type().isa<pir::VectorType>()) {
    size_t shape_size = shape.type().dyn_cast<pir::VectorType>().size();
    vec_shape = std::vector<int64_t>(shape_size, -1);
    *is_from_tensor = true;
  } else if (shape.type().isa<paddle::dialect::DenseTensorType>()) {
    common::DDim shape_dim =
        shape.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    size_t shape_size = common::product(shape_dim);
    if (common::contain_unknown_dim(shape_dim)) {
      shape_size = 1;
    }
    vec_shape = std::vector<int64_t>(shape_size, -1);
    *is_from_tensor = true;
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support VectorType or DenseTensorType "
        "or AllocatedDenseTensorType"));
  }
  return vec_shape;
}

const std::unordered_map<std::string, std::string>& CppTypeToAttrTypeMap() {
  static const std::unordered_map<std::string, std::string> attr_type_map = {
      {"bool", "pir::BoolAttribute"},
      {"int", "pir::Int32Attribute"},
      {"float", "pir::FloatAttribute"},
      {"int64_t", "pir::Int64Attribute"},
      {"std::string", "pir::StrAttribute"},
      {"std::vector<int>", "pir::ArrayAttribute<pir::Int32Attribute>"},
      {"std::vector<float>", "pir::ArrayAttribute<pir::FloatAttribute>"},
      {"std::vector<int64_t>", "pir::ArrayAttribute<pir::Int64Attribute>"},
      {"std::vector<std::string>", "pir::ArrayAttribute<pir::StrAttribute>"}};
  return attr_type_map;
}

const std::unordered_map<std::string, phi::DataType>& StringToDataTypeMap() {
  static std::unordered_map<std::string, phi::DataType> data_type_map{
      {"bool", phi::DataType::BOOL},
      {"uint8", phi::DataType::UINT8},
      {"int8", phi::DataType::INT8},
      {"uint16", phi::DataType::UINT16},
      {"int16", phi::DataType::INT16},
      {"uint32", phi::DataType::UINT32},
      {"int32", phi::DataType::INT32},
      {"uint64", phi::DataType::UINT64},
      {"int64", phi::DataType::INT64},
      {"float32", phi::DataType::FLOAT32},
      {"complex64", phi::DataType::COMPLEX64},
      {"complex128", phi::DataType::COMPLEX128},
      {"Undefined", phi::DataType::UNDEFINED},
      {"pstring", phi::DataType::PSTRING},
      {"float16", phi::DataType::FLOAT16},
      {"bfloat16", phi::DataType::BFLOAT16},
      {"float64", phi::DataType::FLOAT64}};
  return data_type_map;
}

const std::unordered_map<std::string, phi::Place>& StringToPlaceMap() {
  static std::unordered_map<std::string, phi::Place> place_map{
      {"cpu", phi::CPUPlace{}},
      {"gpu", phi::GPUPlace{}},
      {"gpu_pinned", phi::GPUPinnedPlace{}},
      {"xpu", phi::XPUPlace{}},
      {"ipu", phi::IPUPlace{}},
      {":", phi::CustomPlace{}},
      {"undefined", phi::Place{}}};
  return place_map;
}

const std::unordered_map<std::string, phi::DataLayout>&
StringToDataLayoutMap() {
  static std::unordered_map<std::string, phi::DataLayout> data_layout_map{
      {"NHWC", phi::DataLayout::kNHWC},
      {"NCHW", phi::DataLayout::kNCHW},
      {"Undefined", phi::DataLayout::kAnyLayout},
      {"ONEDNN", phi::DataLayout::ONEDNN},
      {"SPARSE_COO", phi::DataLayout::SPARSE_COO},
      {"SPARSE_CSR", phi::DataLayout::SPARSE_CSR},
      {"NDHWC", phi::DataLayout::kNDHWC},
      {"NCDHW", phi::DataLayout::kNCDHW},
      {"PSTRING_UNION", phi::DataLayout::PSTRING_UNION},
      {"STRIDED", phi::DataLayout::STRIDED}};
  return data_layout_map;
}

void SetStopGradient() {}

void SetStopGradient(pir::Value* value) {
  value->set_attribute(
      "stop_gradient",
      pir::BoolAttribute::get(pir::IrContext::Instance(), true));
}

void SetStopGradient(std::vector<pir::Value>* values) {
  for (auto& value : *values) {
    SetStopGradient(&value);
  }
}

void SetStopGradient(paddle::optional<pir::Value>* value) {
  if (value->get_ptr() != nullptr) {
    SetStopGradient(value->get_ptr());
  }
}

void SetStopGradient(paddle::optional<std::vector<pir::Value>>* values) {
  if (values->get_ptr() != nullptr) {
    SetStopGradient(values->get_ptr());
  }
}

void PushStopGradient(const pir::Value& value, std::vector<bool>* arr) {
  if (!IsEmptyValue(value)) {
    arr->push_back(false);
  } else {
    arr->push_back(true);
  }
}

std::vector<std::vector<bool>> ConstructStopGradient(pir::Operation* op) {
  std::vector<std::vector<bool>> stop_gradients(op->results().size());
  for (size_t i = 0; i < op->results().size(); i++) {
    PushStopGradient(op->result(i), &stop_gradients[i]);
  }
  return stop_gradients;
}

bool CanGroupOpRunCpuKernel(const std::vector<::pir::Value>& vec_inputs,
                            const std::vector<::pir::Value>& vec_output) {
  for (size_t i = 0; i < vec_inputs.size(); ++i) {
    auto tmp_in = vec_inputs[i];
    if (!tmp_in || !tmp_in.type()) {
      continue;
    }

    phi::DDim in_dims;

    if (auto type_info =
            tmp_in.type()
                .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()) {
      auto type = tmp_in.type().dyn_cast<AllocatedDenseTensorType>();
      in_dims = type.dims();
      if (type.place().GetType() != phi::AllocationType::CPU) {
        return false;
      }
    } else if (auto type_info =
                   tmp_in.type().dyn_cast<paddle::dialect::DenseTensorType>()) {
      in_dims = type_info.dims();
    }

    // 1. dynamic shape not need lower x86
    if (::common::contain_unknown_dim(in_dims)) {
      return false;
    }
    // 2. size < 4 not need lower x86
    if (phi::product(in_dims) > 4) {
      return false;
    }
  }

  for (size_t i = 0; i < vec_output.size(); ++i) {
    const auto& out = vec_output[i];

    if (!out || !out.type()) {
      continue;
    }

    if (out.type().isa<DenseTensorType>()) {
      auto type = out.type().dyn_cast<DenseTensorType>();

      if (type.dtype().isa<::pir::BFloat16Type>()) {
        return false;
      }

      if (::common::contain_unknown_dim(type.dims())) {
        return false;
      }

      if (phi::product(type.dims()) > 4) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace dialect
}  // namespace paddle
