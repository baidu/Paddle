/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/tensor_meta.h"
<<<<<<< HEAD

#include <utility>
=======
#include "paddle/common/flags.h"
>>>>>>> 8b808f1ca1f3b066d25661279d07b83806836d58
#include "paddle/phi/core/enforce.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

DDim DenseTensorMeta::calc_strides(const DDim& dims) {
  if (dims.size() == -1 || product(dims) <= 0) {
    return dims;
  }

  DDim strides(dims);

  // NOTE: The NHWC and NDHWC in Paddle are implemented by actually modifying
  // the video memory data format, and stride is not required. But it may be
  // used in the future. if (dims.size() == 4 && layout == DataLayout::NHWC) {
  //   strides[1] = 1;
  //   strides[3] = dims[1];
  //   strides[2] = strides[3] * dims[3];
  //   strides[0] = strides[2] * dims[2];
  // } else if (dims.size() == 5 && layout == DataLayout::NDHWC) {
  //   strides[1] = 1;
  //   strides[4] = dims[1];
  //   strides[3] = strides[4] * dims[4];
  //   strides[2] = strides[3] * dims[3];
  //   strides[0] = strides[2] * dims[2];
  // } else {
  //   strides[dims.size() - 1] = 1;
  //   for (int i = dims.size() - 2; i >= 0; --i) {
  //     strides[i] = strides[i + 1] * dims[i + 1];
  //   }
  // }
  auto p_dims = dims.Get();
  auto p_strides = strides.GetMutable();
  switch (dims.size()) {
    case 0:
      return strides;
    case 1:
      p_strides[0] = 1;
      return strides;
    case 2:
      p_strides[1] = 1;
      p_strides[0] = p_dims[1];
      return strides;
    case 3:
      p_strides[2] = 1;
      p_strides[1] = p_dims[2];
      p_strides[0] = p_strides[1] * p_dims[1];
      return strides;
    case 4:
      p_strides[3] = 1;
      p_strides[2] = p_dims[3];
      p_strides[1] = p_strides[2] * p_dims[2];
      p_strides[0] = p_strides[1] * p_dims[1];
      return strides;
    case 5:
      p_strides[4] = 1;
      p_strides[3] = p_dims[4];
      p_strides[2] = p_strides[3] * p_dims[3];
      p_strides[1] = p_strides[2] * p_dims[2];
      p_strides[0] = p_strides[1] * p_dims[1];
      return strides;
    case 6:
      p_strides[5] = 1;
      p_strides[4] = p_dims[5];
      p_strides[3] = p_strides[4] * p_dims[4];
      p_strides[2] = p_strides[3] * p_dims[3];
      p_strides[1] = p_strides[2] * p_dims[2];
      p_strides[0] = p_strides[1] * p_dims[1];
      return strides;
    case 7:
      p_strides[6] = 1;
      p_strides[5] = p_dims[6];
      p_strides[4] = p_strides[5] * p_dims[5];
      p_strides[3] = p_strides[4] * p_dims[4];
      p_strides[2] = p_strides[3] * p_dims[3];
      p_strides[1] = p_strides[2] * p_dims[2];
      p_strides[0] = p_strides[1] * p_dims[1];
      return strides;
    case 8:
      p_strides[7] = 1;
      p_strides[6] = p_dims[7];
      p_strides[5] = p_strides[6] * p_dims[6];
      p_strides[4] = p_strides[5] * p_dims[5];
      p_strides[3] = p_strides[4] * p_dims[4];
      p_strides[2] = p_strides[3] * p_dims[3];
      p_strides[1] = p_strides[2] * p_dims[2];
      p_strides[0] = p_strides[1] * p_dims[1];
      return strides;
    case 9:
      p_strides[8] = 1;
      p_strides[7] = p_dims[8];
      p_strides[6] = p_strides[7] * p_dims[7];
      p_strides[5] = p_strides[6] * p_dims[6];
      p_strides[4] = p_strides[5] * p_dims[5];
      p_strides[3] = p_strides[4] * p_dims[4];
      p_strides[2] = p_strides[3] * p_dims[3];
      p_strides[1] = p_strides[2] * p_dims[2];
      p_strides[0] = p_strides[1] * p_dims[1];
      return strides;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "The rank of input should be less than 9, but received %d.",
          dims.size()));
  }
}

DenseTensorMeta::DenseTensorMeta() { use_gpudnn = true; }

DenseTensorMeta::DenseTensorMeta(DataType dtype, const DDim& dims)
    : dims(dims), dtype(dtype) {
  strides = calc_strides(dims);
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 const DDim& strides)
    : dims(dims), dtype(dtype), strides(strides) {
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout,
                                 size_t offset)
    : dims(dims), dtype(dtype), layout(layout), offset(offset) {
  strides = calc_strides(dims);
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout,
<<<<<<< HEAD
                                 LoD  lod,
                                 size_t offset)
    : dims(dims), dtype(dtype), layout(layout), lod(std::move(lod)), offset(offset) {
=======
                                 const LegacyLoD& legacy_lod,
                                 size_t offset)
    : dims(dims),
      dtype(dtype),
      layout(layout),
      legacy_lod(legacy_lod),
      offset(offset) {
>>>>>>> 337f0d225e59843df33c4ca189ae312b8c36a4f3
  strides = calc_strides(dims);
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(const DenseTensorMeta& other) {
  is_scalar = other.is_scalar;
  use_gpudnn = other.use_gpudnn;
  dims = other.dims;
  dtype = other.dtype;
  layout = other.layout;
  legacy_lod = other.legacy_lod;
  offset = other.offset;
  if (other.strides.size() == -1) {
    strides = calc_strides(dims);
  } else {
    strides = other.strides;
  }
}

DenseTensorMeta& DenseTensorMeta::operator=(const DenseTensorMeta& other) {
  is_scalar = other.is_scalar;
  use_gpudnn = other.use_gpudnn;
  dims = other.dims;
  dtype = other.dtype;
  layout = other.layout;
  legacy_lod = other.legacy_lod;
  offset = other.offset;
  if (other.strides.size() == -1) {
    strides = calc_strides(dims);
  } else {
    strides = other.strides;
  }
  return *this;
}

DenseTensorMeta& DenseTensorMeta::operator=(  // NOLINT
    DenseTensorMeta&& other) {
  is_scalar = other.is_scalar;
  use_gpudnn = other.use_gpudnn;
  dims = other.dims;
  dtype = other.dtype;
  layout = other.layout;
  legacy_lod = std::move(other.legacy_lod);
  offset = other.offset;
  if (other.strides.size() == -1) {
    strides = calc_strides(dims);
  } else {
    strides = other.strides;
  }
  return *this;
}

bool DenseTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (dtype != DataType::UNDEFINED);
  valid = valid && (layout != DataLayout::UNDEFINED);
  valid = valid && (is_scalar || product(dims) >= 0);
  return valid;
}

bool DenseTensorMeta::is_contiguous() const {
  bool is_contiguous = (strides == calc_strides(dims));
  if (!is_contiguous && !FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Not contiguous "
        "Tensor found, something wrong has happened!"));
  }
  return is_contiguous;
}

StringTensorMeta::StringTensorMeta(const DDim& dims) : dims(dims) {}

bool StringTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (is_scalar || product(dims) >= 0);
  return valid;
}

SparseTensorMeta::SparseTensorMeta(const DDim& dims) : dims(dims) {}

SparseTensorMeta::SparseTensorMeta(const DDim& dims, const DataLayout& layout)
    : dims(dims), layout(layout) {}

bool SparseTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (layout != DataLayout::UNDEFINED);
  valid = valid && (product(dims) >= 0);
  return valid;
}

}  // namespace phi
