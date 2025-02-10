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

#include "paddle/phi/core/string_tensor.h"

#include <utility>

#include "glog/logging.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/pstring.h"

namespace phi {

StringTensor::StringTensor() { meta_.offset = 0; }

StringTensor::StringTensor(Allocator* a, const StringTensorMeta& meta)
    : meta_(meta), holder_(a->Allocate(SizeOf(dtype()) * numel())) {
  init_holder();
}

StringTensor::StringTensor(Allocator* a, StringTensorMeta&& meta)
    : meta_(std::move(meta)), holder_(a->Allocate(SizeOf(dtype()) * numel())) {
  init_holder();
}

StringTensor::StringTensor(std::shared_ptr<phi::Allocation>  holder,
                           StringTensorMeta  meta)
    : meta_(std::move(meta)), holder_(std::move(holder)) {}

StringTensor::StringTensor(const StringTensor& other) {  // NOLINT
  this->meta_ = other.meta();
  holder_ = other.holder_;
}

StringTensor& StringTensor::operator=(const StringTensor& other) {
  if (this == &other) return *this;
  meta_ = other.meta();
  holder_ = other.holder_;
  return *this;
}

StringTensor& StringTensor::operator=(  // NOLINT
    StringTensor&& other) noexcept {
  meta_ = std::move(other.meta_);
  std::swap(holder_, other.holder_);
  return *this;
}

int64_t StringTensor::numel() const {
  if (meta_.is_scalar) {
    return 1;
  }
  return product(meta_.dims);
}

bool StringTensor::IsSharedWith(const StringTensor& b) const {
  return holder_ && holder_ == b.holder_;
}

const Place& StringTensor::place() const {
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      common::errors::PreconditionNotMet(
          "Tensor not initialized yet when DenseTensor::place() is called."));
  return holder_->place();
}

const dtype::pstring* StringTensor::data() const {
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      common::errors::PreconditionNotMet(
          "The storage must be valid when call the mutable data function."));
  uintptr_t ptr = reinterpret_cast<uintptr_t>(holder_->ptr()) + meta_.offset;
  return reinterpret_cast<const dtype::pstring*>(ptr);
}

dtype::pstring* StringTensor::data() {
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      common::errors::PreconditionNotMet(
          "The storage must be valid when call the mutable data function."));
  uintptr_t ptr = reinterpret_cast<uintptr_t>(holder_->ptr()) + meta_.offset;
  return reinterpret_cast<dtype::pstring*>(ptr);
}

void StringTensor::set_meta(const StringTensorMeta& meta) {
  PADDLE_ENFORCE_EQ(
      meta.valid(),
      true,
      common::errors::InvalidArgument(
          "Input meta is invalid, please check the meta attribute."));
  meta_.dims = meta.dims;
  meta_.is_scalar = meta.is_scalar;
  meta_.offset = meta.offset;
}

StringTensor& StringTensor::Resize(const DDim& dims) {
  meta_.dims = dims;
  return *this;
}
// TODO(zhoushunjie): need to remove it for general space
void StringTensor::init_holder() {
  void* ptr = holder_->ptr();
  auto& place = holder_->place();
  auto bytes_size = holder_->size();
  VLOG(6) << "Init StringTensor data with bytes:" << bytes_size;
  if (place.GetType() == phi::AllocationType::CPU) {
    std::memset(ptr, 0, bytes_size);
  } else if (place.GetType() == phi::AllocationType::GPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#ifdef PADDLE_WITH_HIP
    hipMemset(ptr, 0, bytes_size);
#else
    cudaMemset(ptr, 0, bytes_size);
#endif
#endif
  } else {
    // TODO(zhoushunjie): Need to support more places
    PADDLE_THROW(
        errors::Unimplemented("StringTensor can only be created in CPU or GPU "
                              "place. But now attempts to "
                              "create StringTensor on %s",
                              place.DebugString()));
  }
}

void* StringTensor::AllocateFrom(Allocator* allocator,
                                 DataType dtype,
                                 size_t requested_size,
                                 bool fake_alloc) {
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      errors::InvalidArgument(
          "Required allocator shall not be nullptr, but received nullptr."));

  size_t bytes = numel() * SizeOf(this->dtype());
  if (fake_alloc) {
    bytes = 0;
  } else {
    PADDLE_ENFORCE_EQ(
        valid(),
        true,
        errors::PreconditionNotMet("The meta data must be valid when call the "
                                   "mutable data function."));
    if (requested_size) {
      PADDLE_ENFORCE_GE(requested_size,
                        bytes,
                        errors::InvalidArgument(
                            "The reserved size %d should be enough to meet the "
                            "volume required by metadata %d.",
                            requested_size,
                            bytes));

      bytes = requested_size;
    }
  }

  if (!holder_ || holder_->size() < bytes + meta_.offset) {
    meta_.offset = 0;
    VLOG(10) << "Allocate string data with bytes: " << bytes;
    holder_.reset();
    holder_ = allocator->Allocate(bytes);
    // Initialize the allocated bytes
    init_holder();
    meta_.offset = 0;
  }
  uintptr_t ptr = reinterpret_cast<uintptr_t>(holder_->ptr()) + meta_.offset;
  return reinterpret_cast<void*>(ptr);
}

dtype::pstring* StringTensor::mutable_data(const phi::Place& place,
                                           size_t requested_size) {
  PADDLE_ENFORCE_GE(
      numel(),
      0,
      common::errors::PreconditionNotMet(
          "The Tensor's element number must be equal or greater than zero. "
          "The Tensor's shape is [",
          dims(),
          "] now"));
  size_t size = numel() * SizeOf(dtype());
  if (requested_size && (requested_size > size)) {
    size = requested_size;
  }

  /* some versions of paddle::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + meta_.offset) {
    holder_.reset();
    holder_ = memory_utils::AllocShared(place, size);
    // Initialize the allocated bytes
    init_holder();
    meta_.offset = 0;
  }
  uintptr_t ptr = reinterpret_cast<uintptr_t>(holder_->ptr()) + meta_.offset;
  return reinterpret_cast<dtype::pstring*>(ptr);
}

}  // namespace phi
