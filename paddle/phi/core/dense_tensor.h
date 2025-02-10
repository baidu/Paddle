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
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/storage_properties.h"
#include "paddle/phi/core/stream.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/utils/test_macros.h"

namespace phi {

class DenseTensorUtils;
namespace distributed {
class DistTensor;
}  // namespace distributed

/// \brief The Dense tensor stores values in a contiguous sequential block
/// of memory where all values are represented. Tensors or multi-dimensional
/// arrays are used in math operators.
/// During the entire life cycle of a DenseTensor, its device type and key
/// metadata are set unchanged.
class TEST_API DenseTensor : public TensorBase,
                             public TypeInfoTraits<TensorBase, DenseTensor> {
 public:
  /// \brief Construct a dense tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of dense tensor.
  DenseTensor(Allocator* a, const DenseTensorMeta& meta);

  /// \brief Construct a dense tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of dense tensor.
  DenseTensor(Allocator* a, DenseTensorMeta&& meta);

  DenseTensor(std::shared_ptr<phi::Allocation>  holder,
              const DenseTensorMeta& meta);

  /// \brief Because dense tensor is a kind of container, we give a default
  /// constructor to use for stl container. But the dense tensor created with
  /// the default constructor is not practical.
  // DenseTensor() = default;

  /// \brief Because dense tensor is a resource handle, we provide a default
  /// move constructor to support move semantics.
  DenseTensor(DenseTensor&& other) = default;

  /// \brief DenseTensor shallow copy constructor.
  DenseTensor(const DenseTensor& other);

  /// \brief DenseTensor shallow copy assignment.
  DenseTensor& operator=(const DenseTensor& other);

  DenseTensor& operator=(DenseTensor&& other) noexcept;

  DenseTensor();

  /// \brief Destroy the tensor object and release exclusive resources.
  virtual ~DenseTensor();

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "DenseTensor"; }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const override;

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const noexcept override { return meta_.dims; }

  /// \brief Returns the stride of the tensor.
  /// \return The stride of the tensor.
  const DDim& strides() const noexcept { return meta_.strides; }

  /// \brief Sets the stride of the tensor.
  /// \param meta The stride of the tensor.
  void set_strides(const DDim& strides) { meta_.strides = strides; }

  /// \brief Returns the lod of the tensor.
  /// \return The lod of the tensor.
  const LegacyLoD& lod() const noexcept { return meta_.legacy_lod; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override { return meta_.dtype; }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override { return meta_.layout; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override;

  /// \brief Returns the meta information of the tensor.
  /// \return The meta information of the tensor.
  const DenseTensorMeta& meta() const noexcept { return meta_; }

  /// \brief Sets the meta information of the tensor. Only when the original
  /// attribute of Tensor is incomplete, can it be reset.
  /// \param meta The meta information of the tensor.
  void set_meta(DenseTensorMeta&& meta);

  void set_meta(const DenseTensorMeta& meta);

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept override { return meta_.valid(); }

  /// \brief Test whether the allocation is allocated.
  /// return Whether the allocation is allocated.
  bool initialized() const override { return holder_ && holder_->ptr(); }

  /// \brief Test whether the holder is created.
  /// \return Whether the holder is created.
  bool has_allocation() const override { return holder_ != nullptr; }

  /// \brief Allocate memory with requested size from allocator.
  /// \return The mutable data pointer value of type T.
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;

  /// \brief Check if allocation is shared with other objects.
  /// \return Whether the allocation is shared with other objects.
  bool IsSharedWith(const DenseTensor& b) const;

  /// \brief Change the shape information in the metadata. If the new size is
  /// larger than the original value, the allocation area will be reallocated.
  /// \param dims The new dims of the dense tensor.
  /// \param lod The new lod of the dense tensor.
  // void Resize(const DDim& dims);
  void ResizeAndAllocate(const DDim& dims);

  DenseTensor& Resize(const DDim& dims);

  /// \brief Change the lod information in the metadata.
  /// \param legacy_lod The new lod of the dense tensor.
  void ResetLoD(const LegacyLoD& legacy_lod);

  /// \brief Returns the actual allocation size occupied by tensor, may be
  /// larger
  /// than its shape dims.
  /// \return The actual allocation size occupied by tensor.
  size_t capacity() const { return holder_->size(); }

  /// \brief Get the const data pointer value of type T.
  /// \return The const data pointer value of type T.
  template <typename T>
  const T* data() const;

  /// \brief Get the const data pointer value of raw type.
  /// \return The const data pointer value of raw type.
  const void* data() const;

  template <typename T>
  T* data();

  void* data();

  /// \brief Get whether the storage_properties is inited.
  /// \return The init status of storage_properties.
  bool storage_properties_initialized() const;

  /// \brief Returns the storage_properties of the tensor.
  /// \return The storage_properties of the tensor.
  template <typename DeviceT>
  const DeviceT& storage_properties() const;

  /// \brief Sets the storage_properties of the tensor.
  /// \param storage_properties The storage_properties of the tensor.
  void set_storage_properties(
      std::unique_ptr<StorageProperties>&& storage_properties);

  void clear() {
    holder_.reset();
    meta_.offset = 0;
  }

 private:
  friend class DenseTensorUtils;
  friend class phi::distributed::DistTensor;

 protected:
  DenseTensorMeta meta_;
  std::shared_ptr<phi::Allocation> holder_;

  /** [ Why need StorageProperties? ]
   *
   * 1. Some hardware or third-party libraries add some additional storage
   * properties on top of the description of the basic DenseTensor, such as
   * memory desc of OneDNN, storage_format and storage_layout of NPU,
   * these members are necessary for optimal performance, but if the properties
   * of each device are added to the DenseTensor with different macro isolation,
   * the memory layout of the DenseTensor will become more fragmented.
   * Under different compilation conditions, the member layout of the
   * DenseTensor is very unstable, which may introduce bugs that are difficult
   * to debug.
   *
   * 2. If the layout of DenseTensor is very different from the framework
   * itself, it is recommended to directly inherit TensorBase to implement
   * SpatialTensor.
   *
   * TODO(chenweihang): merge the dnnl::memory::desc and
   * dnnl::memory::format_tag into StorageProperties, dnnl::memory::desc is a
   * type that takes up a lot of space, original tensor members' size:
   *
   * DenseTensor size: 880
   * -------- ordered members --------:
   * DenseTensorMeta size: 128
   *  - is_scalar_ size: 1
   *  - DDim size: 80
   *  - DataType size: 4
   *  - DataLayout size: 4
   *  - LoD size: 24
   *  - offset size: 8
   *  std::shared_ptr<phi::Allocation> size: 16
   *  std::shared_ptr<InplaceVersion> size: 16 // need to be moved
   *  dnnl::memory::format_tag size: 4 // need to be moved
   *  dnnl::memory::desc size: 696 // need to be moved
   */
  std::unique_ptr<StorageProperties> storage_properties_{nullptr};

 public:
  /* Temporarily put InplaceVersion inside DenseTensor.
  Will move to AutogradMeta as soon as we switch to Eager Dygraph.
  */
  /*
  NOTE(liym27): [ What is TensorInplaceVersion used for? ]

  TensorInplaceVersion is a version counter and every Tensor has a version
  counter. It's used to check whether an inplace operation will result in an
  incorrect gradient calculation. Version is incremented when the data of the
  Variable is modified in place.

  - Question: In what scenarios will version counters be shared?
  - Answer: When two Variables/VarBases share the same C++ Tensor(its Allocation
  may change), both of them share the same version counter. For examples:
   1. `z = paddle.assign(input=x, output=y)`, `z` shares the same version
  counter of `y` because z and y is the same VarBase;
   2. `y = x.detach()`, `y` shares the same version counter of `x`.

  - Question: In what scenarios will version counters NOT be shared?
  - Answer: Replacing a `Variable`'s data by calling
  `Tensor::ShareDataWith(...)` or `Tensor::ShareBufferWith(...)`. Because they
  share the same Allocation but not phi::DenseTensor.

  - Question: Why put the inplace_version_counter_ in phi::DenseTensor instead
  of Allocation or Variable?
  - Answer:
   1. Tensor can call ResetHolder() to reset the corresponding Allocation so
  that the inplace_version_counter_ changes if it's in Allocation, which will
  lead to confusing information about inplace version.
   2. If inplace_version_counter_ is in Variable, different VariableWrappers
   should be able to share the same Variable. However, a VariableWrapper hold a
   Variable object but not a pointer.
 */
  class InplaceVersion {
   public:
    bool IsUnique() const { return inplace_version_ == 0; }
    void Bump() { ++inplace_version_; }
    uint32_t CurrentVersion() const { return inplace_version_; }
    void SetInplaceVersionToZero() { inplace_version_ = 0; }

   private:
    uint32_t inplace_version_{0};
  };

 protected:
  std::shared_ptr<InplaceVersion> inplace_version_counter_ =
      std::make_shared<InplaceVersion>();

#ifndef PADDLE_WITH_CUSTOM_KERNEL
#include "paddle/phi/core/dense_tensor.inl"
#endif
};

}  // namespace phi
