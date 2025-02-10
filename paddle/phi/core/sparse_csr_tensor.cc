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

#include "paddle/phi/core/sparse_csr_tensor.h"

#include <utility>

namespace phi {

SparseCsrTensor::SparseCsrTensor() {
  DenseTensor crows, cols, values;
  this->non_zero_crows_ = crows;
  this->non_zero_cols_ = cols;
  this->non_zero_elements_ = values;
}

inline void check_shape(const DDim& dims) {
  bool valid = dims.size() == 2 || dims.size() == 3;

  PADDLE_ENFORCE_EQ(
      valid,
      true,
      common::errors::InvalidArgument("the SparseCsrTensor only support 2-D or "
                                      "3-D Tensor, but get %d-D Tensor",
                                      dims.size()));
}
#define Check(non_zero_crows, non_zero_cols, non_zero_elements, dims)          \
  {                                                                            \
    check_shape(dims);                                                         \
    PADDLE_ENFORCE_EQ(                                                         \
        non_zero_cols.place(),                                                 \
        non_zero_crows.place(),                                                \
        common::errors::InvalidArgument(                                       \
            "non_zero_crows and non_zero_cols must have the same place."));    \
    PADDLE_ENFORCE_EQ(                                                         \
        non_zero_cols.place(),                                                 \
        non_zero_elements.place(),                                             \
        common::errors::InvalidArgument(                                       \
            "non_zero_cols and non_zero_elements must have the same place.")); \
  }

SparseCsrTensor::SparseCsrTensor(const DenseTensor& non_zero_crows,
                                 DenseTensor  non_zero_cols,
                                 const DenseTensor& non_zero_elements,
                                 const DDim& dims)
    : non_zero_crows_(non_zero_crows),
      non_zero_cols_(std::move(non_zero_cols)),
      non_zero_elements_(non_zero_elements) {
  if (non_zero_crows.initialized()) {
    Check(non_zero_crows_, non_zero_cols_, non_zero_elements_, dims);
  } else {
    // create a empty tensor
    check_shape(dims);
  }
  meta_.dims = dims;
  meta_.layout = DataLayout::NCHW;
  meta_.dtype = non_zero_elements.dtype();
}

SparseCsrTensor::SparseCsrTensor(const SparseCsrTensor& other) {  // NOLINT
  this->non_zero_crows_ = other.non_zero_crows_;
  this->non_zero_cols_ = other.non_zero_cols_;
  this->non_zero_elements_ = other.non_zero_elements_;
  set_meta(other.meta());
}

SparseCsrTensor& SparseCsrTensor::operator=(const SparseCsrTensor& other) {
  this->non_zero_crows_ = other.non_zero_crows();
  this->non_zero_cols_ = other.non_zero_cols();
  this->non_zero_elements_ = other.non_zero_elements();
  set_meta(other.meta());
  return *this;
}

void* SparseCsrTensor::AllocateFrom(Allocator* allocator,
                                    DataType dtype,
                                    size_t requested_size,
                                    bool fake_alloc) {
  return non_zero_elements_.AllocateFrom(
      allocator, dtype, requested_size, fake_alloc);
}

void SparseCsrTensor::set_type(const DataType dtype) { meta_.dtype = dtype; }

void SparseCsrTensor::set_layout(const DataLayout layout) {
  meta_.layout = layout;
}

void SparseCsrTensor::Resize(const DDim& dense_dims,
                             const int64_t non_zero_num) {
  PADDLE_ENFORCE_EQ(
      this->initialized(),
      true,
      common::errors::InvalidArgument(
          "the SparseCsrTensor must be initialized when call Resize "
          "function."));
  check_shape(dense_dims);

  int64_t crows_size = dense_dims[0] + 1;
  if (dense_dims.size() == 3) {
    // batch_size = dims[0]
    crows_size = dense_dims[0] * (dense_dims[1] + 1);
  }

  DDim crows_dims = common::make_ddim({crows_size});
  this->non_zero_crows_.Resize(crows_dims);

  DDim col_dims = common::make_ddim({non_zero_num});
  this->non_zero_cols_.Resize(col_dims);
  this->non_zero_elements_.Resize(col_dims);
}

void SparseCsrTensor::SetMember(const DenseTensor& non_zero_crows,
                                const DenseTensor& non_zero_cols,
                                const DenseTensor& non_zero_elements,
                                const DDim& dims) {
  Check(non_zero_crows, non_zero_cols, non_zero_elements, dims);
  this->non_zero_crows_ = non_zero_crows;
  this->non_zero_cols_ = non_zero_cols;
  this->non_zero_elements_ = non_zero_elements;
  meta_.dims = dims;
}

void SparseCsrTensor::SetMember(const DenseTensor& non_zero_crows,
                                const DenseTensor& non_zero_cols,
                                const DenseTensor& non_zero_elements,
                                const SparseTensorMeta& meta) {
  Check(non_zero_crows, non_zero_cols, non_zero_elements, meta.dims);
  this->non_zero_crows_ = non_zero_crows;
  this->non_zero_cols_ = non_zero_cols;
  this->non_zero_elements_ = non_zero_elements;
  set_meta(meta);
}

void SparseCsrTensor::set_meta(SparseTensorMeta&& meta) {
  PADDLE_ENFORCE_EQ(meta_.valid(),
                    false,
                    common::errors::InvalidArgument(
                        "Only when the original attribute of Tensor is "
                        "incomplete, can it be reset."));
  meta_ = std::move(meta);
}

void SparseCsrTensor::set_meta(const SparseTensorMeta& meta) {
  PADDLE_ENFORCE_EQ(
      meta.valid(),
      true,
      common::errors::InvalidArgument(
          "Input meta is invalid, please check the meta attribute."));
  meta_.dims = meta.dims;
  meta_.dtype = meta.dtype;
  meta_.layout = meta.layout;
}

int32_t SparseCsrTensor::sparse_dim() const { return 2; }

int32_t SparseCsrTensor::dense_dim() const {
  int32_t nze_dim = this->non_zero_elements_.dims().size();
  int32_t batch_dim = this->non_zero_crows_.dims().size() - 1;
  // layout of SparseCsrTensor has not been implemented yet
  // int32_t block_dim =  = (layout_ == kSparseBsr || layout_ == kSparseBsc ? 2
  // : 0);
  int32_t block_dim = 0;
  return nze_dim - batch_dim - block_dim - 1;
}
}  // namespace phi
