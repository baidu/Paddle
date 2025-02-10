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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {

class DenseTensorUtils;

/// \brief The SparseCsrTensor uses three 1-D DenseTensors to represent
/// the row index , column index and non zero elements of the original
/// DenseTensor.
/// where non_zero_crows_ represents the compressed row index,
/// non_zero_cols_ represents the column index of non zero elements in original
/// DenseTensor,
/// non_zero_elements_ represents the non zero elements of original DenseTensor.
class SparseCsrTensor : public TensorBase,
                        public TypeInfoTraits<TensorBase, SparseCsrTensor> {
 public:
  SparseCsrTensor();
  /// \brief Because sparse csr tensor is a resource handle, we provide a
  /// default
  /// move constructor to support move semantics.
  SparseCsrTensor(SparseCsrTensor&& other) = default;

  /// \brief SparseCsrTensor shallow copy constructor.
  SparseCsrTensor(const SparseCsrTensor& other);

  /// \brief create the sparse csr tensor.
  /// \param non_zero_crows The compressed row index of non zero elements in
  /// original dense tensor.
  /// \param non_zero_cols The column index of non zero elements in original
  /// dense tensor.
  /// \param non_zero_elements The non zero elements of original dense tensor.
  /// \param dims The dims of original dense tensor.
  SparseCsrTensor(const DenseTensor& non_zero_crows,
                  DenseTensor  non_zero_cols,
                  const DenseTensor& non_zero_elements,
                  const DDim& dims);

  /// \brief SparseCsrTensor shallow copy assignment.
  SparseCsrTensor& operator=(const SparseCsrTensor& other);

  /// \brief Destroy the tensor object and release exclusive resources.
  virtual ~SparseCsrTensor() = default;

  /// \brief This function is not recommended
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "SparseCsrTensor"; }

  /// \brief Returns the compressed row index of non zero elements in original
  /// dense tensor.
  /// \return The compressed row index of non zero elements in original dense
  /// tensor.
  const DenseTensor& crows() const { return non_zero_crows_; }

  /// Note: This function will removed soon. It is recommended to use crows()
  const DenseTensor& non_zero_crows() const { return non_zero_crows_; }

  /// \brief Returns the column index of non zero elements in original dense
  /// tensor.
  /// \return The column index of non zero elements in original dense tensor.
  const DenseTensor& cols() const { return non_zero_cols_; }

  /// Note: This function will removed soon. It is recommended to use cols()
  const DenseTensor& non_zero_cols() const { return non_zero_cols_; }

  /// \brief Returns the non zero elements in original dense tensor.
  /// \return The non zero elements in original dense tensor.
  const DenseTensor& values() const { return non_zero_elements_; }

  /// Note: This function will removed soon. It is recommended to use indices()
  const DenseTensor& non_zero_elements() const { return non_zero_elements_; }

  /// \brief Returns the total number of non zero elements in original dense
  /// tensor.
  int64_t nnz() const { return non_zero_elements_.numel(); }

  /// \brief Return the number of elements contained in original dense tensor
  /// \return The number of elements contained in original dense tensor
  int64_t numel() const override { return product(meta_.dims); }

  /// \brief Returns the dims of the original dense tensor.
  /// \return The dims of the original dense tensor.
  const DDim& dims() const noexcept override { return meta_.dims; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override { return meta_.dtype; }

  /// \brief get the sparse dim
  /// \return The sparse dim of the tensor.
  int32_t sparse_dim() const;

  /// \brief get the dense dim
  /// \return The dense dim of the tensor.
  int32_t dense_dim() const;

#ifndef PADDLE_WITH_CUSTOM_KERNEL
  void set_type(const DataType dtype);
#endif

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override { return meta_.layout; }

#ifndef PADDLE_WITH_CUSTOM_KERNEL
  void set_layout(const DataLayout layout);
#endif

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override { return non_zero_elements_.place(); }

  /// \brief Test whether the non_zero_elements_ metadata is valid.
  /// \return Whether the non_zero_elements_ metadata is valid.
  bool valid() const noexcept override { return non_zero_elements_.valid(); }

  /// \brief Test whether the holder is created.
  /// \return Whether the holder is created.
  bool has_allocation() const override { return values().has_allocation(); }

  /// \brief Test whether the non_zero_elements_ storage is allocated.
  /// In special cases, when nnz=0, non_zero_elements_ will not need to be
  /// initialized, but it is necessary to return true here, otherwise the
  /// gradient will be None. return Whether the non_zero_elements_ storage is
  /// allocated.
  bool initialized() const override {
    return values().initialized() || (nnz() == 0 && numel() > 0);
  }

  /// \brief resize sparse csr tensor.
  /// \param dense_dims The dims of original dense tensor.
  /// \param non_zero_num The total number of non zero element
  void Resize(const DDim& dense_dims, const int64_t non_zero_num);

  /// \brief set the member of sparse csr tensor.
  /// \param non_zero_crows The compressed row index of non zero elements in
  /// original dense tensor.
  /// \param non_zero_cols The column index of non zero elements in original
  /// dense tensor.
  /// \param non_zero_elements The non zero elements of original dense tensor.
  /// \param dims The dims of original dense tensor.
  void SetMember(const DenseTensor& non_zero_crows,
                 const DenseTensor& non_zero_cols,
                 const DenseTensor& non_zero_elements,
                 const DDim& dims);

  /// \brief set the member of sparse csr tensor.
  /// \param non_zero_crows The compressed row index of non zero elements in
  /// original dense tensor.
  /// \param non_zero_cols The column index of non zero elements in original
  /// dense tensor.
  /// \param non_zero_elements The non zero elements of original dense tensor.
  /// \param meta The meta of original dense tensor.
  void SetMember(const DenseTensor& non_zero_crows,
                 const DenseTensor& non_zero_cols,
                 const DenseTensor& non_zero_elements,
                 const SparseTensorMeta& meta);

  /// \brief Get a mutable pointer of non_zero_crows.
  /// return a mutable pointer of non_zero_crows.
  DenseTensor* mutable_crows() { return &non_zero_crows_; }

  /// Note: This function will removed soon. It is recommended to use
  /// mutable_crows()
  DenseTensor* mutable_non_zero_crows() { return &non_zero_crows_; }

  /// \brief Get a mutable pointer of non_zero_cols.
  /// return a mutable pointer of non_zero_cols.
  DenseTensor* mutable_cols() { return &non_zero_cols_; }

  /// Note: This function will removed soon. It is recommended to use
  /// mutable_cols()
  DenseTensor* mutable_non_zero_cols() { return &non_zero_cols_; }

  /// \brief Get a mutable pointer of non_zero_elements.
  /// return a mutable pointer of non_zero_elements.
  DenseTensor* mutable_values() { return &non_zero_elements_; }

  /// Note: This function will removed soon. It is recommended to use
  /// mutable_values()
  DenseTensor* mutable_non_zero_elements() { return &non_zero_elements_; }

  /// \brief Returns the meta information of the tensor.
  /// \return The meta information of the tensor.
  const SparseTensorMeta& meta() const noexcept { return meta_; }

  void set_meta(SparseTensorMeta&& meta);

  void set_meta(const SparseTensorMeta& meta);

  /// \brief set the dims of original dense tensor
  void set_dims(const DDim& dims) { meta_.dims = dims; }

 protected:
  SparseTensorMeta meta_;

 private:
  friend class DenseTensorUtils;
  // save the compressed rows information of non zero elements
  DenseTensor non_zero_crows_;
  // save the columns information of non zero elements
  DenseTensor non_zero_cols_;
  // save the non zero elements
  DenseTensor non_zero_elements_;
  /* --------------------------- */
  /*   example: 2-D Tensor */
  /* --------------------------- */
  /*
     x = [[0, 1, 0, 0],
          [2, 0, 0, 3],
          [0, 0, 4, 0],
          [0, 5, 0, 6]]
     dims_ = (4, 4)
     non_zero_elements_ = [1, 2, 3, 4, 5, 6]
     non_zero_crows_ = [0, 1, 3, 4, 6]
     non_zero_cols_ = [1, 0, 3, 2, 1, 3]
   */

  /* --------------------------- */
  /*   example: 3-D Tensor */
  /*   the non zero elements of different batch will be concat together */
  /* --------------------------- */
  /*
     x = [[[0, 1, 0, 0],
          [2, 0, 0, 3],
          [0, 0, 4, 0],
          [0, 5, 0, 6]],
         [[0, 1, 0, 0],
          [2, 0, 0, 3],
          [0, 0, 4, 0],
          [0, 5, 0, 0]]]
     dims_ = (2, 4, 4)
     non_zero_elements_ = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5]
     non_zero_crows_ = [0, 1, 3, 4, 6, 0, 1, 2, 4, 5]
     non_zero_cols_ = [1, 0, 3, 2, 1, 3, 1, 0, 3, 2, 1]
   */
};

}  // namespace phi
