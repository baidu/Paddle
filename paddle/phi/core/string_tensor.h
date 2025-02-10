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

#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {

namespace dtype {
class pstring;
}  // namespace dtype

/// \brief In Paddle 2.3, we add a new type of Tensor, StringTensor,
/// which is designed for string data management.
/// During the entire life cycle of a StringTensor, its device type and key
/// metadata are set unchanged.
class StringTensorUtils;

class StringTensor : public TensorBase,
                     public TypeInfoTraits<TensorBase, StringTensor> {
 public:
  /// \brief Construct a string tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of string tensor.
  StringTensor(Allocator* a, const StringTensorMeta& meta);

  /// \brief Construct a string tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of string tensor.
  StringTensor(Allocator* a, StringTensorMeta&& meta);

  StringTensor(std::shared_ptr<phi::Allocation>  holder,
               StringTensorMeta  meta);

  /// \brief Because string tensor is a resource handle, we provide a default
  /// move constructor to support move semantics.
  StringTensor(StringTensor&& other) = default;

  StringTensor(const StringTensor& other);

  StringTensor();
  /// \brief StringTensor shallow copy assignment.
  StringTensor& operator=(const StringTensor& other);

  StringTensor& operator=(StringTensor&& other) noexcept;
  /// \brief Destroy the tensor object and release exclusive resources.
  virtual ~StringTensor() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "StringTensor"; }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const override;

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const noexcept override { return meta_.dims; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override;

  /// \brief Returns the meta information of the tensor.
  /// \return The meta information of the tensor.
  const StringTensorMeta& meta() const noexcept { return meta_; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override { return DataType::PSTRING; }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override {
    return DataLayout::PSTRING_UNION;
  }

  void set_meta(const StringTensorMeta& meta);

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept override { return meta_.valid(); }

  /// \brief Test whether the holder is created.
  /// \return Whether the holder is created.
  bool has_allocation() const override { return holder_ != nullptr; }

  /// \brief Test whether the storage is allocated.
  /// return Whether the storage is allocated.
  bool initialized() const override { return holder_ && holder_->ptr(); }

  /// \brief Check if storage is shared with other objects.
  /// \return Whether the storage is shared with other objects.
  bool IsSharedWith(const StringTensor& b) const;

  StringTensor& Resize(const DDim& dims);

  /// \brief Returns the actual storage size occupied by tensor, may be larger
  /// than its shape dims.
  /// \return The actual storage size occupied by tensor.
  size_t capacity() const { return holder_->size(); }

  /// \brief Get the const data pointer value of pstring type.
  /// \return The const data pointer value of pstring type.
  const dtype::pstring* data() const;
  dtype::pstring* data();

  void clear() {
    holder_.reset();
    meta_.offset = 0;
  }
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;
  dtype::pstring* mutable_data(const phi::Place& place,
                               size_t requested_size = 0);

 private:
  friend class StringTensorUtils;

 private:
  StringTensorMeta meta_;
  std::shared_ptr<phi::Allocation> holder_;
  void init_holder();
};

}  // namespace phi
