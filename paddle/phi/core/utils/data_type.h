/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <iostream>
#include <map>
#include <string>
#include <typeindex>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
namespace phi {

#define _PhiForEachDataTypeHelper_(callback, cpp_type, data_type) \
  callback(cpp_type, data_type);

#define _PhiForEachDataType_(callback)                                 \
  _PhiForEachDataTypeHelper_(callback, float, DataType::FLOAT32);      \
  _PhiForEachDataTypeHelper_(                                          \
      callback, ::phi::dtype::float16, DataType::FLOAT16);             \
  _PhiForEachDataTypeHelper_(                                          \
      callback, ::phi::dtype::bfloat16, DataType::BFLOAT16);           \
  _PhiForEachDataTypeHelper_(                                          \
      callback, ::phi::dtype::float8_e4m3fn, DataType::FLOAT8_E4M3FN); \
  _PhiForEachDataTypeHelper_(                                          \
      callback, ::phi::dtype::float8_e5m2, DataType::FLOAT8_E5M2);     \
  _PhiForEachDataTypeHelper_(callback, double, DataType::FLOAT64);     \
  _PhiForEachDataTypeHelper_(callback, int, DataType::INT32);          \
  _PhiForEachDataTypeHelper_(callback, int64_t, DataType::INT64);      \
  _PhiForEachDataTypeHelper_(callback, bool, DataType::BOOL);          \
  _PhiForEachDataTypeHelper_(callback, uint8_t, DataType::UINT8);      \
  _PhiForEachDataTypeHelper_(callback, int16_t, DataType::INT16);      \
  _PhiForEachDataTypeHelper_(callback, int8_t, DataType::INT8);        \
  _PhiForEachDataTypeHelper_(                                          \
      callback, ::phi::dtype::complex<float>, DataType::COMPLEX64);    \
  _PhiForEachDataTypeHelper_(                                          \
      callback, ::phi::dtype::complex<double>, DataType::COMPLEX128);

#define _PhiForEachDataTypeTiny_(callback)                    \
  _PhiForEachDataTypeHelper_(callback, int, DataType::INT32); \
  _PhiForEachDataTypeHelper_(callback, int64_t, DataType::INT64);

template <typename Visitor>
inline void VisitDataType(phi::DataType type, Visitor visitor) {
#define PhiVisitDataTypeCallback(cpp_type, data_type) \
  do {                                                \
    if (type == data_type) {                          \
      visitor.template apply<cpp_type>();             \
      return;                                         \
    }                                                 \
  } while (0)

  _PhiForEachDataType_(PhiVisitDataTypeCallback);
#undef PhiVisitDataTypeCallback
  PADDLE_THROW(common::errors::Unimplemented(
      "Not supported phi::DataType(%d) as data type.", static_cast<int>(type)));
}

template <typename Visitor>
inline void VisitDataTypeTiny(phi::DataType type, Visitor visitor) {
#define PhiVisitDataTypeCallbackTiny(cpp_type, data_type) \
  do {                                                    \
    if (type == data_type) {                              \
      visitor.template apply<cpp_type>();                 \
      return;                                             \
    }                                                     \
  } while (0)

  _PhiForEachDataTypeTiny_(PhiVisitDataTypeCallbackTiny);
#undef PhiVisitDataTypeCallbackTiny
  PADDLE_THROW(common::errors::Unimplemented(
      "Not supported phi::DataType(%d) as data type.", static_cast<int>(type)));
}

inline bool IsComplexType(const DataType& type) {
  return (type == DataType::COMPLEX64 || type == DataType::COMPLEX128);
}

inline DataType ToComplexType(const DataType& type) {
  switch (type) {
    case DataType::FLOAT32:
      return DataType::COMPLEX64;
    case DataType::FLOAT64:
      return DataType::COMPLEX128;
    default:
      PADDLE_THROW(errors::Unimplemented(
          "Can not transform data type (%s) to complex type, now only support "
          "float32 and float64 real value.",
          type));
  }
}

inline DataType ToRealType(const DataType& type) {
  switch (type) {
    case DataType::COMPLEX64:
      return DataType::FLOAT32;
    case DataType::COMPLEX128:
      return DataType::FLOAT64;
    default:
      PADDLE_THROW(errors::Unimplemented(
          "Can not transform data type (%s) to real type, now only support "
          "complex64 and complex128 value.",
          type));
  }
}

// In some cases we need to use the conversion between phi::DataType and
// fluid proto::VarType::Type, but can't depend on the proto::VarType::Type.
// So here we defined an enum type ProtoDataType which corresponds to
// proto::VarType::Type in fluid, but keeps only the data types we need.
// Note: The ProtoDataType (defined here) and proto::VarType::Type (defined
// in framework.pb.h) need to be modified simultaneously.
enum ProtoDataType {
  BOOL = 0,
  INT16 = 1,
  INT32 = 2,
  INT64 = 3,
  FP16 = 4,
  FP32 = 5,
  FP64 = 6,
  RAW = 17,
  UINT8 = 20,
  INT8 = 21,
  BF16 = 22,
  COMPLEX64 = 23,
  COMPLEX128 = 24,
  PSTRING = 29,
  FP8_E4M3FN = 32,
  FP8_E5M2 = 33
};

inline DataType TransToPhiDataType(const int& dtype) {
  // Set the order of case branches according to the frequency with
  // the data type is used
  switch (dtype) {
    case ProtoDataType::FP32:
      return DataType::FLOAT32;
    case ProtoDataType::FP64:
      return DataType::FLOAT64;
    case ProtoDataType::INT64:
      return DataType::INT64;
    case ProtoDataType::INT32:
      return DataType::INT32;
    case ProtoDataType::INT8:
      return DataType::INT8;
    case ProtoDataType::UINT8:
      return DataType::UINT8;
    case ProtoDataType::INT16:
      return DataType::INT16;
    case ProtoDataType::COMPLEX64:
      return DataType::COMPLEX64;
    case ProtoDataType::COMPLEX128:
      return DataType::COMPLEX128;
    case ProtoDataType::FP8_E4M3FN:
      return DataType::FLOAT8_E4M3FN;
    case ProtoDataType::FP8_E5M2:
      return DataType::FLOAT8_E5M2;
    case ProtoDataType::FP16:
      return DataType::FLOAT16;
    case ProtoDataType::BF16:
      return DataType::BFLOAT16;
    case ProtoDataType::BOOL:
      return DataType::BOOL;
    case ProtoDataType::PSTRING:
      return DataType::PSTRING;
    case ProtoDataType::RAW:
      return DataType::ALL_DTYPE;
    default:
      return DataType::UNDEFINED;
  }
}

inline int TransToProtoVarType(const DataType& dtype) {
  // Set the order of case branches according to the frequency with
  // the data type is used
  switch (dtype) {
    case DataType::FLOAT32:
      return ProtoDataType::FP32;
    case DataType::FLOAT64:
      return ProtoDataType::FP64;
    case DataType::INT64:
      return ProtoDataType::INT64;
    case DataType::INT32:
      return ProtoDataType::INT32;
    case DataType::INT8:
      return ProtoDataType::INT8;
    case DataType::UINT8:
      return ProtoDataType::UINT8;
    case DataType::INT16:
      return ProtoDataType::INT16;
    case DataType::COMPLEX64:
      return ProtoDataType::COMPLEX64;
    case DataType::COMPLEX128:
      return ProtoDataType::COMPLEX128;
    case DataType::FLOAT8_E4M3FN:
      return ProtoDataType::FP8_E4M3FN;
    case DataType::FLOAT8_E5M2:
      return ProtoDataType::FP8_E5M2;
    case DataType::FLOAT16:
      return ProtoDataType::FP16;
    case DataType::BFLOAT16:
      return ProtoDataType::BF16;
    case DataType::BOOL:
      return ProtoDataType::BOOL;
    case DataType::PSTRING:
      return ProtoDataType::PSTRING;
    case DataType::UNDEFINED:
      return ProtoDataType::RAW;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported data type `%s` when casting it into "
          "paddle data type.",
          dtype));
  }
}

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
inline ncclDataType_t ToNCCLDataType(DataType type) {
  if (type == DataType::FLOAT32) {
    return ncclFloat;
  } else if (type == DataType::FLOAT64) {
    return ncclDouble;
  } else if (type == DataType::INT32) {
    return ncclInt;
  } else if (type == DataType::INT64) {
    return ncclInt64;
  } else if (type == DataType::FLOAT16) {
    return ncclFloat16;
  } else if (type == DataType::UINT8) {
    return ncclUint8;
  } else if (type == DataType::INT8) {
    return ncclInt8;
  } else if (type == DataType::BOOL) {
    return ncclUint8;
#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
  } else if (type == DataType::BFLOAT16) {
    return ncclBfloat16;
#endif
  } else {
    PADDLE_THROW(
        errors::Unimplemented("This datatype in nccl is not supported."));
  }
}
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
inline BKCLDataType ToBKCLDataType(DataType type) {
  if (type == DataType::FLOAT32) {
    return BKCL_FLOAT;
  } else if (type == DataType::FLOAT64) {
    return BKCL_FLOAT64;
  } else if (type == DataType::INT32) {
    return BKCL_INT32;
  } else if (type == DataType::INT64) {
    return BKCL_INT64;
  } else if (type == DataType::FLOAT16) {
    return BKCL_FLOAT16;
  } else if (type == DataType::UINT8) {
    return BKCL_UINT8;
  } else if (type == DataType::BOOL) {
    return BKCL_UINT8;
  } else if (type == DataType::BFLOAT16) {
    return BKCL_BFLOAT16;
  } else {
    PADDLE_THROW(
        errors::Unimplemented("This datatype in bkcl is not supported."));
  }
}
#endif

#if defined(PADDLE_WITH_UCC)
#include <ucc/api/ucc.h>
inline ucc_datatype_t ToUCCDataType(DataType type) {
  if (type == DataType::FLOAT32) {
    return UCC_DT_FLOAT32;
  } else if (type == DataType::FLOAT64) {
    return UCC_DT_FLOAT64;
  } else if (type == DataType::INT32) {
    return UCC_DT_INT32;
  } else if (type == DataType::INT64) {
    return UCC_DT_INT64;
  } else if (type == DataType::FLOAT16) {
    return UCC_DT_FLOAT16;
  } else if (type == DataType::UINT8) {
    return UCC_DT_UINT8;
  } else if (type == DataType::INT8) {
    return UCC_DT_INT8;
  } else if (type == DataType::BOOL) {
    return UCC_DT_UINT8;
  } else if (type == DataType::BFLOAT16) {
    return UCC_DT_BFLOAT16;
  } else {
    PADDLE_THROW(
        errors::Unimplemented("This datatype in ucc is not supported."));
  }
}
#endif

}  // namespace phi
