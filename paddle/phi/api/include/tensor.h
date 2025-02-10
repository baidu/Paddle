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

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
using gpuStream_t = hipStream_t;
#endif

#include "paddle/common/layout.h"
#include "paddle/common/macros.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"

namespace phi {
class DenseTensor;
class TensorBase;
}  // namespace phi

namespace common {
class DDim;
}  // namespace common

namespace paddle {
// TODO(chenweihang): Remove the experimental namespace for Scalar and IntArray
using Scalar = experimental::Scalar;
using IntArray = experimental::IntArray;

class AbstractAutogradMeta {
 public:
  // No AbstractAutogradMeta should be created
  virtual const paddle::Tensor& fw_grad(uint64_t level,
                                        const paddle::Tensor& self) const = 0;
  virtual const paddle::Tensor& fw_grad(uint64_t level) const = 0;
  virtual void set_fw_grad(const paddle::Tensor& new_grad,
                           const paddle::Tensor& self_base,
                           uint64_t level,
                           bool is_inplace_op) = 0;
  virtual ~AbstractAutogradMeta() = default;
};

/**
 * Tensor is the API description of the basic data structure in the
 * [ "Paddle HIgh reusability operator (phi)" Library ].
 *
 * It is not limited to a simple n-dimensional array.
 * It contains a smart pointer to `TensorImpl`. The data description contained
 * in Tensor is defined by TensorImpl. Tensor only defines the interface for
 * computation.
 *
 * Note: Tensor can be NULL state, Tensor is meaningful only when the
 * TensorImpl to which it is pointed is not empty.
 *
 * Note: For the consistency of C++ API self, and the consistency between C++
 * API and Python API, all member methods of Tensor are named with lowercase
 * letters and underscores.
 *
 * Note: Tensor cannot be inherited. The heterogeneous Tensor implementation
 * can be achieved by inheriting the underlying TensorBase.
 *
 * Note: This Tensor API is suitable for training，inference and custom
 * operators.
 */

class PADDLE_API Tensor final {
 public:
  /* Part 1: Construction and destruction methods */

  /**
   * @brief Construct a new Tensor object
   */
  Tensor() = default;

  /**
   * @brief Construct a new Tensor object by copy
   */
  Tensor(const Tensor&) = default;

  /**
   * @brief Construct a new Tensor object by move
   */
  Tensor(Tensor&&) noexcept = default;

  /**
   * @brief Construct a new Tensor object by a TensorBase pointer
   *
   * @param tensor_impl
   */
  explicit Tensor(std::shared_ptr<phi::TensorBase> tensor_impl);

  /**
   * @brief Construct a new Tensor object on the target place.
   *
   * This is a deprecated method and may be removed in the future!!!
   *
   * @param place
   */
  explicit Tensor(const Place& place);

  /**
   * @brief Construct a new Tensor object on the target place
   * with specified shape.
   *
   * This is a deprecated method and may be removed in the future!!!
   *
   * @param place
   * @param shape
   */
  Tensor(const Place& place, const std::vector<int64_t>& shape);

  /**
   * @brief Construct a new Tensor object by a TensorBase pointer and name
   *
   * @param tensor_impl
   */
  Tensor(std::shared_ptr<phi::TensorBase> tensor_impl, const std::string& name);

  /**
   * @brief Construct a new Tensor object with name
   *
   * @note Internal method, used to adapt original execution mechanism and
   * debug analysis in the development of new dygraph. It may be removed in
   * the future.
   * */
  explicit Tensor(const std::string& name) : name_(name) {}

  /**
   * @brief Construct a new Tensor object by a TensorBase pointer, autograd meta
   * and name
   *
   * @param tensor_impl
   * @param autograd_meta
   * @param name
   */
  Tensor(std::shared_ptr<phi::TensorBase> tensor_impl,
         std::shared_ptr<AbstractAutogradMeta> autograd_meta,
         const std::string& name);

  /* Part 2: Dimension, DataType and DataLayout methods */

  /**
   * @brief Return the number of elements of Tensor.
   *
   * @return int64_t
   */
  int64_t numel() const;

  /**
   * @brief Get the size of current tensor.
   *
   * The compatible method of `Tensor::numel()`.
   * This is a deprecated method and may be removed in the future!
   *
   * @return int64_t
   */
  int64_t size() const;

  /**
   * @brief Return the dimensions of Tensor.
   *
   * @return common::DDim
   */
  const common::DDim& dims() const;

  /**
   * @brief Return the shape (dimensions) of Tensor.
   *
   * The compatible method of `Tensor::dims()`.
   * This is a deprecated method and may be removed in the future!
   *
   * @return std::vector<int64_t>
   */
  std::vector<int64_t> shape() const;

  /**
   * @brief Return the strides (dimensions) of Tensor.
   *
   * @return common::DDim
   */
  const common::DDim& strides() const;

  /**
   * @brief Reset the shape of the tensor.
   * @note: This method means Reset the shape of the tensor,
   * and must be called before calling mutable_data() or
   * copy_to(const Place& place), this is not a standard definition of
   * reshape behavior, so we will deprecated this feature in the future.
   *
   * @param shape
   */
  void reshape(const std::vector<int64_t>& shape);

  /**
   * @brief Return the data type of Tensor.
   *
   * @return DataType
   */
  DataType dtype() const;

  /**
   * @brief Return the data type of Tensor.
   *
   * The compatible method of `Tensor::dtype()`.
   * This is a deprecated method and may be removed in the future!
   *
   * @return DataType
   */
  DataType type() const;

  /**
   * @brief Return the layout of Tensor.
   *
   * @return DataLayout
   */
  phi::DataLayout layout() const;

  /**
   * @brief Determine whether tensor is DenseTensor
   *
   * @return bool
   */
  bool is_dense_tensor() const;

  /**
   * @brief Determine whether tensor is DistTensor
   *
   * @return bool
   */
  bool is_dist_tensor() const;

  /**
   * @brief Determine whether tensor is SelectedRows
   *
   * @return bool
   */
  bool is_selected_rows() const;

  /**
   * @brief Determine whether tensor is SparseCooTensor
   *
   * @return bool
   */
  bool is_sparse_coo_tensor() const;

  /**
   * @brief Determine whether tensor is SparseCsrTensor
   *
   * @return bool
   */
  bool is_sparse_csr_tensor() const;

  /**
   * @brief Determine whether tensor is StringTensor
   *
   * @return bool
   */
  bool is_string_tensor() const;

  /* Part 3: Device and Backend methods */

  /**
   * @brief Return the place (device) of Tensor.
   *
   * @return Place
   */
  const Place& place() const;

  /**
   * @brief Determine whether the tensor device is CPU
   *
   * @return bool
   */
  bool is_cpu() const;

  /**
   * @brief Determine whether the tensor device is GPU
   *
   * @return bool
   */
  bool is_gpu() const;

  /**
   * @brief Determine whether the tensor device is GPU_PINNED
   *
   * @return bool
   */
  bool is_gpu_pinned() const;

  /**
   * @brief Determine whether the tensor device is XPU
   *
   * @return bool
   */
  bool is_xpu() const;

  /**
   * @brief Determine whether the tensor device is CustomDevice
   *
   * @return bool
   */
  bool is_custom_device() const;

  /* Part 4: Data Access methods */

  /**
   * @brief Get the memory pointer in CPU or GPU with specific data type.
   * It's usually used to get the output data pointer, same as the T* data().
   *
   * @tparam T
   * @return T*
   */
  template <typename T>
  T* mutable_data();

  /**
   * @brief Get the memory pointer in CPU or GPU with specific data type.
   *
   * It's usually used to get the output data pointer.
   * This is a deprecated method and may be removed in the future!
   *
   * @tparam T
   * @param place
   * @return T*
   */
  template <typename T>
  T* mutable_data(const Place& place);

  /**
   * @brief Get the const memory pointer directly.
   * It's usually used to get the output data pointer.
   *
   * @tparam T
   * @return T*
   */
  template <typename T>
  const T* data() const;

  /**
   * @brief Get the memory pointer directly.
   * It's usually used to get the mutable output data pointer.
   *
   * @tparam T
   * @return T*
   */
  template <typename T>
  T* data();

  /**
   * @brief Get the const memory pointer directly.
   * It's usually used to get the output data pointer.
   *
   * @tparam T
   * @return T*
   */
  const void* data() const;

  /**
   * @brief Get the memory pointer directly.
   * It's usually used to get the mutable output data pointer.
   *
   * @tparam T
   * @return T*
   */
  void* data();

  /**
   * @brief Return a sub-tensor of the given tensor.
   * It is usually used to extract a sub-tensor (which supports
   * modifying the data of the original tensor) to perform further
   * operations.
   *
   * @param begin_idx The index of the start row (inclusive) to slice.
   *                  The index number begins from 0.
   * @param end_idx The index of the end row (exclusive) to slice.
   *                 The index number begins from begin_idx + 1.
   * @return Tensor
   */
  Tensor slice(int64_t begin_idx, int64_t end_idx) const;

  /**
   * @brief Return the implementation of current Tensor.
   *
   * @return std::shared_ptr<phi::TensorBase>
   */
  const std::shared_ptr<phi::TensorBase>& impl() const;

  /**
   * @brief Set the implementation of current Tensor.
   *
   * @param impl
   */
  void set_impl(const std::shared_ptr<phi::TensorBase>& impl);

  /**
   * @brief Set the implementation of current Tensor.
   *
   * @param impl
   */
  void set_impl(std::shared_ptr<phi::TensorBase>&& impl);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  /**
   * @brief Get the stream where the tensor is currently located
   * This is a deprecated method and may be removed in the future!
   *
   * @return gpuStream_t
   */
  gpuStream_t stream() const;
#endif

  /**
   * @brief Return the name of Tensor.
   * @note Used to adapt original execution mechanism and debug analysis
   * in the development of new dygraph.
   *
   * @return const std::string&
   */
  const std::string& name() const;

  /**
   * @brief Set name of Tensor.
   * @note Used to adapt original execution mechanism and debug analysis
   * in the development of new dygraph.
   *
   * @param const std::string& name
   */
  void set_name(const std::string& name);

  /* Part 5: Data Transform methods */
  /* Alert!!!!: All copy method can only deep copy impl, autograd info only be
   * copied */
  /* out of phi */
  /**
   * @brief Copy the current Tensor data to the specified device
   * and return the new Tensor. It's usually used to set the input tensor data.
   * @note The Tensor's `copy_to` method is deprecated since version 2.3, and
   * will be removed in version 2.4, please use `copy_to` method without
   * template argument instead.
   * reason: copying a Tensor to another device does not need to specify the
   * data type template argument
   *
   * @tparam T
   * @param target_place The target place of which the tensor will copy to.
   * @return Tensor
   */
  template <typename T>
  Tensor copy_to(const Place& target_place) const;

  /**
   * @brief Transfer the current Tensor to the specified device and return.
   *
   * @param place The target place of which the tensor will copy to.
   * @param blocking Should we copy this in sync way.
   * @return Tensor
   */
  Tensor copy_to(const Place& place, bool blocking) const;

  /**
   * @brief Transfer the source Tensor to current Tensor.
   *
   * @param src The source Tensor to be copied.
   * @param blocking Should we copy this in sync way.
   * @return void
   */
  void copy_(const Tensor& src, const Place& target_place, bool blocking);

  /**
   * @brief Cast datatype from one to another
   *
   * @param target_type
   * @return Tensor
   */
  Tensor cast(DataType target_type) const;

  /* Part 6: Status utils methods */

  /**
   * @brief Determine whether it is a meaningful Tensor
   *
   * @return bool
   */
  bool defined() const;

  /**
   * @brief Determine whether Tensor has allocation
   *
   * @return bool
   */
  bool has_allocation() const;

  /**
   * @brief Determine whether Tensor is initialized.
   *
   * @return bool
   */
  bool initialized() const;

  /**
   * @brief Determine whether Tensor is initialized.
   * This is a deprecated method and may be removed in the future!
   *
   * @return bool
   */
  bool is_initialized() const;

  /**
   * @brief Reset the Tensor implementation
   */
  void reset();

  /* Part 7: Operator overloading */

  /**
   * @brief Assignment operator
   *
   * @param x
   * @return Tensor&
   */
  Tensor& operator=(const Tensor& x) &;

  /**
   * @brief Move assignment operator
   *
   * @param x
   * @return Tensor&
   */
  Tensor& operator=(Tensor&& x) & noexcept;

  /**
   * @brief Tensor operants
   *
   * @param other
   * @return Tensor
   */
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;
  Tensor operator+(const Scalar& other) const;
  Tensor operator-(const Scalar& other) const;
  Tensor operator*(const Scalar& other) const;
  Tensor operator/(const Scalar& other) const;
  Tensor operator<(const Tensor& other) const;
  Tensor operator<=(const Tensor& other) const;
  Tensor operator==(const Tensor& other) const;
  Tensor operator!=(const Tensor& other) const;
  Tensor operator>(const Tensor& other) const;
  Tensor operator>=(const Tensor& other) const;
  Tensor operator-() const;
  Tensor operator~() const;
  Tensor operator&(const Tensor& other) const;
  Tensor operator|(const Tensor& other) const;
  Tensor operator^(const Tensor& other) const;

  /* Part 8: Autograd methods */

  /**
   * @brief Get the autograd meta object pointer
   *
   * @return AbstractAutogradMeta*
   */
  AbstractAutogradMeta* get_autograd_meta() const;

  /**
   * @brief Get the shared pointer of autograd meta object
   *
   * @return std::shared_ptr<AbstractAutogradMeta>&
   */
  const std::shared_ptr<AbstractAutogradMeta>& mutable_autograd_meta() const;

  /**
   * @brief Set the autograd meta object
   *
   * @param autograd_meta
   */
  void set_autograd_meta(std::shared_ptr<AbstractAutogradMeta> autograd_meta);

  /* Part 9: Inplace methods */

  /**
   * @brief Increase inplace version
   */
  void bump_inplace_version();

  /**
   * @brief Get current inplace version
   *
   * @return uint32_t
   */
  uint32_t current_inplace_version();

  /**
   * @brief Reset inplace version
   */
  void reset_inplace_version(bool set_to_zero = false);

  /* Part 10: Auto generated Tensor methods */

  /* Part 11: Methods of converting underlying TensorType to each other
   */
  /**
   * @brief Convert DenseTensor or SparseCsrTensor to SparseCooTensor
   *
   * @param sparse_dim The number of sparse dimensions
   * @return Tensor
   */
  Tensor to_sparse_coo(const int64_t sparse_dim) const;

  /**
   * @brief Convert DenseTensor or SparseCooTensor to SparseCsrTensor
   *
   * @return Tensor
   */
  Tensor to_sparse_csr() const;

  /**
   * @brief Convert SparseCooTensor or SparseCsrTensor to DenseTensor
   *
   * @return Tensor
   */
  Tensor to_dense() const;

  /* Part 12: Contiguous methods */

  /**
   * @brief Determine whether tensor is contiguous
   *
   * @return bool
   */
  bool is_contiguous() const;

  /**
   * @brief Returns a contiguous in memory tensor containing the same data as
   * current Tensor. If self tensor is already contiguous, this function returns
   * the current Tensor.
   *
   * @return Tensor
   */
  Tensor contiguous();

  /* Part 13: Forward AD related*/
  // The Forward AD API functions below are low level and are not to be used by
  // end users who should use the API provided in torch/csrc/autograd.h

  const Tensor& _fw_grad(uint64_t level) const;

  void _set_fw_grad(const Tensor& new_grad, uint64_t level, bool is_inplace_op);

  std::shared_ptr<phi::TensorBase> get_impl() const;

  const Tensor& _fw_primal(int64_t level) const;

 private:
  /**
   * [ Why use abstract TensorImpl interface here? ]
   *
   * We hope that the data structure at the API level of the framework can be
   * unified to Tensor, but Tensor itself is heterogeneous.
   *
   * Tensor can generally be represented by void* and size_t, place.
   * This is suitable for most scenarios including CPU, GPU, HIP, etc.,
   * but there are a few cases where this definition cannot be described,
   * such as the Tensor representation in third-party lib such as Metal,
   * OpenCL, etc., as well as some special Tensor implementations, including
   * Tensor containing only one Scalar value, or Tensor representing String,
   * etc.
   *
   * Therefore, we hope to use a unified interface to shield the underlying
   * heterogeneous Tensor implementation, so that the API level can be unified
   * to one `Tensor`.
   */
  std::shared_ptr<phi::TensorBase> impl_{nullptr};

  /**
   * [ Why need abstract AbstractAutogradMeta here? ]
   *
   * Dynamic graphs need to hold backward information
   *
   * [ Why AutogradMeta not in TensorImpl? ]
   *
   * 1. AutogradMeta is only used in dynamic graph, It is execution-related
   *    information, not Tensor data description-related information.
   * 2. Kernel calculation does not require AutogradMeta.
   */
  std::shared_ptr<AbstractAutogradMeta> autograd_meta_{nullptr};

  /**
   * Tensor name: used to adapt original execution mechanism and debug analysis
   * in the development of new dygraph.
   */
  std::string name_{""};

 public:
  // Tensor C++ APIs
  // Example: Tensor add(const Tensor& other) const;
  Tensor add(const Tensor& y) const;
  Tensor divide(const Tensor& y) const;
  Tensor multiply(const Tensor& y) const;
  Tensor subtract(const Tensor& y) const;
  Tensor add(const Scalar& y) const;
  Tensor divide(const Scalar& y) const;
  Tensor multiply(const Scalar& y) const;
  Tensor subtract(const Scalar& y) const;
  Tensor less_equal(const Tensor& y) const;
  Tensor less_than(const Tensor& y) const;
  Tensor equal(const Tensor& y) const;
  Tensor not_equal(const Tensor& y) const;
  Tensor greater_equal(const Tensor& y) const;
  Tensor greater_than(const Tensor& y) const;
  Tensor bitwise_and(const Tensor& y) const;
  Tensor bitwise_or(const Tensor& y) const;
  Tensor bitwise_xor(const Tensor& y) const;
  Tensor bitwise_not() const;
  Tensor pow(const Tensor& y) const;
  Tensor pow(const Scalar& y) const;

  Tensor exp() const;
  Tensor floor() const;
  Tensor gather_nd(const Tensor& index) const;
  Tensor log() const;
  Tensor roll(const IntArray& shifts = {},
              const std::vector<int64_t>& axis = {}) const;
  Tensor scatter(const Tensor& index,
                 const Tensor& updates,
                 bool overwrite = true) const;
  Tensor scatter_nd_add(const Tensor& index, const Tensor& updates) const;
  Tensor abs() const;
  Tensor assign() const;
  Tensor elementwise_pow(const Tensor& y) const;
  Tensor expand(const IntArray& shape) const;
  Tensor matmul(const Tensor& y,
                bool transpose_x = false,
                bool transpose_y = false) const;
  Tensor max(const IntArray& axis = {}, bool keepdim = false) const;
  Tensor maximum(const Tensor& y) const;
  Tensor minimum(const Tensor& y) const;
  Tensor scale(const Scalar& scale = 1.0,
               const Scalar& bias = 0.0,
               bool bias_after_scale = true) const;
  Tensor sum(const IntArray& axis = {},
             DataType dtype = DataType::UNDEFINED,
             bool keepdim = false) const;
  Tensor tile(const IntArray& repeat_times = {}) const;
};

PADDLE_API Tensor operator+(const Scalar& x, const Tensor& y);

PADDLE_API Tensor operator-(const Scalar& x, const Tensor& y);

PADDLE_API Tensor operator*(const Scalar& x, const Tensor& y);

PADDLE_API Tensor operator/(const Scalar& x, const Tensor& y);

}  // namespace paddle
