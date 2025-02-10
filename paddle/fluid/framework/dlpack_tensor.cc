// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/dlpack_tensor.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/utils/visit_place.h"

namespace paddle::framework::internal {
template <typename T>
static ::DLDataType GetDLDataTypeCode() {
  ::DLDataType dtype;
  if (std::is_same<T, phi::dtype::complex<float>>::value ||
      std::is_same<T, phi::dtype::complex<double>>::value) {
    dtype.code = kDLComplex;
  } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
    dtype.code = kDLBfloat;
  } else if (std::is_same<T, phi::dtype::float16>::value ||
             std::is_floating_point<T>::value) {
    dtype.code = kDLFloat;
  } else if (std::is_same<T, bool>::value) {
    // Since std::is_unsigned<bool>::value is True,
    // it is necessary to evaluate bool before std::is_unsigned.
    dtype.code = kDLBool;
  } else if (std::is_unsigned<T>::value) {
    dtype.code = kDLUInt;
  } else if (std::is_integral<T>::value) {
    dtype.code = kDLInt;
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "Unsupported data type (%s), only supports float16, float, unsigned "
        "int and int.",
        common::demangle(typeid(T).name())));
  }
  dtype.bits = 8 * sizeof(T);
  dtype.lanes = 1;
  return dtype;
}

static std::unordered_map<int, ::DLDataType> CreateDLDataTypeMap() {
  static std::unordered_map<int, ::DLDataType> result;

#define REG_DL_DATA_TYPE(cpp_type, proto_type) \
  result[static_cast<int>(proto_type)] = GetDLDataTypeCode<cpp_type>()

  _ForEachDataType_(REG_DL_DATA_TYPE);
#undef REG_DL_DATA_TYPE
  return result;
}

static DLDataType GetDLDataTypeFromTypeIndex(proto::VarType::Type type) {
  static auto type_to_dtype_map = CreateDLDataTypeMap();
  static auto type_to_dtype_map_end_it = type_to_dtype_map.end();
  auto it = type_to_dtype_map.find(static_cast<int>(type));
  PADDLE_ENFORCE_NE(it,
                    type_to_dtype_map_end_it,
                    common::errors::InvalidArgument(
                        "Unsupported data type (%s).", DataTypeToString(type)));
  return it->second;
#undef REG_DL_DATA_TYPE
}

struct DLDeviceVisitor {
  using argument_type = const phi::Place &;
  using result_type = ::DLDevice;
  inline ::DLDevice operator()(const phi::CPUPlace &place) const {
    ::DLDevice device;
    device.device_type = kDLCPU;
    device.device_id = 0;
    return device;
  }

  inline ::DLDevice operator()(const phi::IPUPlace &place) const {
    PADDLE_THROW(
        common::errors::Unimplemented("phi::IPUPlace is not supported"));
  }

  inline ::DLDevice operator()(const phi::XPUPlace &place) const {
    PADDLE_THROW(
        common::errors::Unimplemented("phi::XPUPlace is not supported"));
  }

  inline ::DLDevice operator()(const phi::CustomPlace &place) const {
    PADDLE_THROW(
        common::errors::Unimplemented("phi::CustomPlace is not supported"));
  }

  inline ::DLDevice operator()(const phi::GPUPlace &place) const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    ::DLDevice device;
    device.device_type = kDLCUDA;
    device.device_id = place.device;  // NOLINT
    return device;
#else
    PADDLE_THROW(common::errors::Unavailable(
        "phi::GPUPlace is not supported in CPU only version."));
#endif
  }

  inline ::DLDevice operator()(const phi::GPUPinnedPlace &place) const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    ::DLDevice device;
    device.device_type = kDLCUDAHost;
    device.device_id = 0;
    return device;
#else
    PADDLE_THROW(common::errors::Unavailable(
        "phi::GPUPinnedPlace is not supported in CPU only version."));
#endif
  }
};
}  // namespace paddle::framework::internal
namespace paddle::framework {

struct PaddleDLMTensor {
  phi::DenseTensor handle;
  DLManagedTensor tensor;
};

static void deleter(DLManagedTensor *self) {
  if (self && self->manager_ctx) {
    delete[] self->dl_tensor
        .shape;  // delete shape allocated in toDLPack manually
    delete[] self->dl_tensor
        .strides;  // delete strides allocated in toDLPack manually
    delete static_cast<PaddleDLMTensor *>(self->manager_ctx);
  }
}

DLManagedTensor *toDLPack(const phi::DenseTensor &src) {
  PaddleDLMTensor *pdDLMTensor(new PaddleDLMTensor);
  pdDLMTensor->handle = const_cast<phi::DenseTensor &>(src);
  pdDLMTensor->tensor.manager_ctx = pdDLMTensor;
  pdDLMTensor->tensor.deleter = &deleter;

  // init ndim
  using DimType = decltype(pdDLMTensor->tensor.dl_tensor.ndim);  // int32_t
  auto _shape = src.dims();
  pdDLMTensor->tensor.dl_tensor.ndim = static_cast<DimType>(_shape.size());
  DimType ndim = pdDLMTensor->tensor.dl_tensor.ndim;

  // init shape
  int64_t *shape = new int64_t[ndim];
  for (DimType i = 0; i < ndim; ++i) {
    shape[i] = _shape[i];
  }
  pdDLMTensor->tensor.dl_tensor.shape = shape;

  // init strides
  auto _strides = src.strides();
  int64_t *strides = new int64_t[ndim];
  for (int i = 0; i < src.dims().size(); i++) {
    strides[i] = _strides[i];
    if (shape[i] < 2) {
      strides[i] = 1;
    }
  }
  pdDLMTensor->tensor.dl_tensor.strides = strides;

  pdDLMTensor->tensor.dl_tensor.data = const_cast<void *>(src.data());
  auto place = src.place();
  pdDLMTensor->tensor.dl_tensor.device =
      phi::VisitPlace(place, internal::DLDeviceVisitor());
  pdDLMTensor->tensor.dl_tensor.dtype = internal::GetDLDataTypeFromTypeIndex(
      framework::TransToProtoVarType(src.dtype()));
  pdDLMTensor->tensor.dl_tensor.byte_offset = 0;
  return &(pdDLMTensor->tensor);
}

DLPackTensor::DLPackTensor(const phi::DenseTensor &tensor, LaneType lanes)
    : t_{}, shape_{} {
  // init data, data buffer
  t_.data = const_cast<void *>(tensor.data());

  // init device, DLDevice type with device_type and device_id
  auto place = tensor.place();
  t_.device = phi::VisitPlace(place, internal::DLDeviceVisitor());

  // init dtype
  t_.dtype = internal::GetDLDataTypeFromTypeIndex(
      framework::TransToProtoVarType(tensor.dtype()));
  t_.dtype.lanes = lanes;

  // init ndim, tensor rank
  auto &dims = tensor.dims();
  using DimType = decltype(t_.ndim);  // int
  t_.ndim = static_cast<DimType>(dims.size());

  // init shape, tensor dims
  t_.shape = shape_;
  for (DimType i = 0; i < t_.ndim; ++i) {
    t_.shape[i] = dims[i];
  }

  // init strides, nullptr means the tensor is compact
  t_.strides = nullptr;

  // init byte_offset
  t_.byte_offset = 0;
}

::DLManagedTensor *DLPackTensor::ToDLManagedTensor() {
  // init shape
  auto shape = new int64_t[t_.ndim];
  using DimType = decltype(t_.ndim);  // int
  for (DimType i = 0; i < t_.ndim; ++i) {
    shape[i] = t_.shape[i];
  }
  t_.shape = shape;

  // init strides
  auto strides = new int64_t[t_.ndim];
  for (DimType i = 0; i < t_.ndim; ++i) {
    strides[i] = 1;
  }
  for (DimType i = t_.ndim - 2; i >= 0; --i) {
    strides[i] = t_.shape[i + 1] * strides[i + 1];
  }
  t_.strides = strides;

  auto tensor = new DLManagedTensor;
  tensor->dl_tensor = t_;

  tensor->deleter = [](DLManagedTensor *arg) {
    delete[] arg->dl_tensor.shape;
    delete[] arg->dl_tensor.strides;
    delete arg;
  };

  tensor->manager_ctx = nullptr;

  return tensor;
}

}  // namespace paddle::framework
