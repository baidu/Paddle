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

#pragma once

#include <memory>

#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/core/tensor_status.h"

namespace paddle {
namespace memory {
namespace allocation {
class Allocation;
}
}
}

namespace pten {

using DataType = paddle::experimental::DataType;

/**
 * The implementation of general Tensor (For CPU, CUDA, HIP, etc.), similar
 * to the Tensor in fluid, contains a pointer to Allocation and a series of
 * descriptive metadata and status required by Tensor.
 *
 * DenseTensor is still a base class, it may have inherited classes.
 *
 * The memory layout of these inherited classes is consistent with the
 * basic DenseTensor, except that a small number of members are added to
 * further specialize the description of the tensor.
 *
 * If the memory layout is different, it cannot be described based on the
 * general Allocation, and it needs to be directly inherited from
 * TensorBase.
 */
class DenseTensor : public TensorBase {
 public:
  // Not allowed to initialize a tensor without descriptive metadata
  DenseTensor() = delete;

  // DenseTensor(const DenseTensor&) = delete;
  // DenseTensor& operator=(const DenseTensor&) = delete;
  DenseTensor(DenseTensor&&) = delete;
  DenseTensor& operator=(DenseTensor&&) = delete;

  /**
   * If we still malloc memory by mutable_data,
   * the DenseTensor doesn't need complicated constructor.
   *
   * Note: Tensor objects lacking meta information are not allowed to exist.
   */
  DenseTensor(const DenseTensorMeta& meta, const TensorStatus& status)
      : meta_(meta), status_(status) {
    int64_t init_numel = paddle::framework::product(meta.dims);
    if (init_numel >= 0) {
      numel_ = init_numel;
    }
  }

  DenseTensor(const DenseTensorMeta& meta,
              const TensorStatus& status,
              Backend backend)
      : meta_(meta), status_(status), backend_(backend) {
    int64_t init_numel = paddle::framework::product(meta.dims);
    if (init_numel >= 0) {
      numel_ = init_numel;
    }
  }

  DenseTensor(DenseTensorMeta&& meta, TensorStatus&& status)
      : meta_(std::move(meta)), status_(std::move(status)) {
    int64_t init_numel = paddle::framework::product(meta.dims);
    if (init_numel >= 0) {
      numel_ = init_numel;
    }
  }

  int64_t numel() const override { return numel_; }

  const paddle::framework::DDim& dims() const override { return meta_.dims; }

  DataType data_type() const override { return meta_.type; }

  DataLayout layout() const override { return meta_.layout; }

  const paddle::platform::Place& place() const override;

  Backend backend() const override { return backend_; }

  bool valid() const override { return allocation_ != nullptr; }

  bool initialized() const override { return allocation_ != nullptr; }

  /* member methods */

  const std::shared_ptr<paddle::memory::allocation::Allocation>& allocation()
      const {
    return allocation_;
  }

  const DenseTensorMeta& meta() const { return meta_; }

  DenseTensorMeta* mutable_meta() { return &meta_; }

  /* Data Access Methods */

  const void* data() const;

  void* mutable_data();

  template <typename T>
  const T* data() const {
    static_assert(std::is_pod<T>::value || std::is_same<T, void>::value,
                  "T must be POD when call Tensor.data<T>().");
    return reinterpret_cast<const T*>(data());
  }

  // NOTE: mutable_data does not hold arguments. Before calling mutable_data,
  // please make sure that Tensor has maintained
  // the correct meta and status.
  //
  // TODO(chenweihang): We need to be able to specify the allocator when
  // mutable_data, or directly remove the mutable_data method.
  // DenseTensor cannot actively apply for memory. Its memory application is
  // handled by the DeviceContext->AllocateTensorData interface.
  // I prefer the latter
  template <typename T>
  T* mutable_data() {
    static_assert(std::is_pod<T>::value,
                  "T must be POD when call Tensor.mutable_data<T>().");
    return reinterpret_cast<T*>(mutable_data());
  }

  // For non-API and non-member interfaces, we still follow the C++ code style?

  void Resize(const DDim& dims) { meta_.dims = dims; }

  void ShareAllocation(const std::shared_ptr<
                       paddle::memory::allocation::Allocation>& allocation);

  paddle::platform::Place GetPlaceByBackend() const;

  size_t MemorySize() const;

  void CheckMemorySize() const;

 private:
  // The actual Tensor storage holder
  std::shared_ptr<paddle::memory::allocation::Allocation> allocation_;
  // The Tensor meta data
  DenseTensorMeta meta_;
  // The Tensor status data
  TensorStatus status_;

  Backend backend_{Backend::CPU};
  int64_t numel_{1};
  size_t offset_{0};
};

}  // namespace pten
