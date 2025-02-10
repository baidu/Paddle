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

#include "paddle/phi/api/include/tensor.h"

#include <memory>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "paddle/common/ddim.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
Tensor singleton_undefined_tensor;

using DeviceContextPool = experimental::DeviceContextPool;
using DefaultAllocator = experimental::DefaultAllocator;

/////// Tensor Methods ////////

/* Part 1: Construction and destruction methods */

Tensor::Tensor(std::shared_ptr<phi::TensorBase> tensor_impl)
    : impl_(std::move(tensor_impl)) {
  PADDLE_ENFORCE_NOT_NULL(impl_,
                          common::errors::InvalidArgument(
                              "TensorImpl with nullptr is not supported"));
}

Tensor::Tensor(std::shared_ptr<phi::TensorBase> tensor_impl,
               std::shared_ptr<AbstractAutogradMeta> autograd_meta,
               const std::string &name)
    : impl_(std::move(tensor_impl)),
      autograd_meta_(std::move(autograd_meta)),
      name_(name) {
  PADDLE_ENFORCE_NOT_NULL(impl_,
                          common::errors::InvalidArgument(
                              "TensorImpl with nullptr is not supported"));
}

Tensor::Tensor(const Place &place) {
  LOG_FIRST_N(WARNING, 1)
      << "The Tensor(place) constructor is deprecated since version "
         "2.3, and will be removed in version 2.4! Please use "
         "`paddle::empty/full` method to create a new "
         "Tensor instead. "
         "Reason: A legal tensor cannot be constructed only based on "
         "the `place`, and datatype, shape, layout, etc. is also "
         "required.";
  DefaultAllocator alloc(place);
  impl_ = std::make_shared<phi::DenseTensor>(
      &alloc,
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           common::make_ddim({}),
                           phi::DataLayout::NCHW));
}

Tensor::Tensor(const Place &place, const std::vector<int64_t> &shape) {
  LOG_FIRST_N(WARNING, 1)
      << "The Tensor(place, shape) constructor is deprecated since "
         "version 2.3, and will be removed in version 2.4! Please use "
         "`paddle::empty/full` method to create a new "
         "Tensor instead. "
         "Reason: A legal tensor cannot be constructed only based on "
         "the `place` and `shape`, and datatype, layout, etc. is also "
         "required.";
  DefaultAllocator alloc(place);
  impl_ = std::make_shared<phi::DenseTensor>(
      &alloc,
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           common::make_ddim({shape}),
                           phi::DataLayout::NCHW));
}

Tensor::Tensor(std::shared_ptr<phi::TensorBase> tensor_impl,
               const std::string &name)
    : impl_(std::move(tensor_impl)), name_(name) {}

/* Part 2: Dimension, DataType and DataLayout methods */

int64_t Tensor::numel() const { return impl_->numel(); }

int64_t Tensor::size() const { return impl_->numel(); }

const phi::DDim &Tensor::dims() const { return impl_->dims(); }

std::vector<int64_t> Tensor::shape() const {
  const auto &dims = impl_->dims();
  return common::vectorize<int64_t>(dims);
}

const phi::DDim &Tensor::strides() const {
  if (is_dense_tensor()) {
    return static_cast<phi::DenseTensor *>(impl_.get())->strides();
  } else if (is_dist_tensor()) {
    return static_cast<phi::distributed::DistTensor *>(impl_.get())
        ->value()
        .strides();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support strides operation on DenseTensor and DistTensor now."));
  }
}

void Tensor::reshape(const std::vector<int64_t> &shape) {
  LOG_FIRST_N(WARNING, 1)
      << "The function of resetting the shape of the uninitialized "
         "Tensor of the `reshape` method is deprecated since version "
         "2.3, and will be removed in version 2.4, please use "
         "`paddle::empty/full` method to create a new Tensor "
         "instead. "
         "reason: `reshape` means changing the tensor shape without "
         "touching underlying data, this requires the total size of "
         "the tensor to remain constant.";
  if (is_dense_tensor()) {
    static_cast<phi::DenseTensor *>(impl_.get())
        ->Resize(common::make_ddim(shape));
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support reshape operation on DenseTensor now."));
  }
}

DataType Tensor::dtype() const { return impl_->dtype(); }

DataType Tensor::type() const { return impl_->dtype(); }

phi::DataLayout Tensor::layout() const { return impl_->layout(); }

bool Tensor::is_dense_tensor() const {
  return phi::DenseTensor::classof(impl_.get());
}
bool Tensor::is_dist_tensor() const {
  if (impl_ == nullptr) {
    return false;
  }
  return phi::distributed::DistTensor::classof(impl_.get());
}
bool Tensor::is_selected_rows() const {
  return phi::SelectedRows::classof(impl_.get());
}
bool Tensor::is_sparse_coo_tensor() const {
  return phi::SparseCooTensor::classof(impl_.get());
}
bool Tensor::is_sparse_csr_tensor() const {
  return phi::SparseCsrTensor::classof(impl_.get());
}
bool Tensor::is_string_tensor() const {
  return phi::StringTensor::classof(impl_.get());
}
/* Part 3: Device and Backend methods */

const Place &Tensor::place() const {
  PADDLE_ENFORCE_NOT_NULL(
      impl_,
      common::errors::PermissionDenied(
          "Null pointer error, the impl_ of Tensor should not be "
          "Null when calling Tensor::place()."));
  return impl_->place();
}

bool Tensor::is_cpu() const {
  return place().GetType() == phi::AllocationType::CPU;
}

bool Tensor::is_gpu() const {
  return place().GetType() == phi::AllocationType::GPU;
}

bool Tensor::is_gpu_pinned() const {
  return place().GetType() == phi::AllocationType::GPUPINNED;
}

bool Tensor::is_xpu() const {
  return place().GetType() == phi::AllocationType::XPU;
}

bool Tensor::is_custom_device() const {
  return place().GetType() == phi::AllocationType::CUSTOM;
}

/* Part 4: Data Access methods */

template <typename T>
T *Tensor::mutable_data() {
  LOG_FIRST_N(WARNING, 1)
      << "Allocating memory through `mutable_data` method is "
         "deprecated since version 2.3, and `mutable_data` method "
         "will be removed in version 2.4! Please use "
         "`paddle::empty/full` method to create a new "
         "Tensor with allocated memory, and use data<T>() method "
         "to get the memory pointer of tensor instead. "
         "Reason: When calling `mutable_data` to allocate memory, "
         "the place, datatype, and data layout of tensor may be in "
         "an illegal state.";
  if (is_dense_tensor()) {
    return static_cast<phi::DenseTensor *>(impl_.get())
        ->mutable_data<T>(place());
  }
  return nullptr;
}

template PADDLE_API bool *Tensor::mutable_data<bool>();
template PADDLE_API int8_t *Tensor::mutable_data<int8_t>();
template PADDLE_API uint8_t *Tensor::mutable_data<uint8_t>();
template PADDLE_API int16_t *Tensor::mutable_data<int16_t>();
template PADDLE_API uint16_t *Tensor::mutable_data<uint16_t>();
template PADDLE_API int32_t *Tensor::mutable_data<int32_t>();
template PADDLE_API uint32_t *Tensor::mutable_data<uint32_t>();
template PADDLE_API int64_t *Tensor::mutable_data<int64_t>();
template PADDLE_API uint64_t *Tensor::mutable_data<uint64_t>();
template PADDLE_API phi::dtype::bfloat16 *
Tensor::mutable_data<phi::dtype::bfloat16>();
template PADDLE_API phi::dtype::float16 *
Tensor::mutable_data<phi::dtype::float16>();
template PADDLE_API float *Tensor::mutable_data<float>();
template PADDLE_API double *Tensor::mutable_data<double>();
template PADDLE_API phi::dtype::complex<float>
    *Tensor::mutable_data<phi::dtype::complex<float>>();
template PADDLE_API phi::dtype::complex<double>
    *Tensor::mutable_data<phi::dtype::complex<double>>();
template PADDLE_API phi::dtype::float8_e4m3fn *
Tensor::mutable_data<phi::dtype::float8_e4m3fn>();
template PADDLE_API phi::dtype::float8_e5m2 *
Tensor::mutable_data<phi::dtype::float8_e5m2>();

template <typename T>
T *Tensor::mutable_data(const Place &place) {
  LOG_FIRST_N(WARNING, 1)
      << "Allocating memory through `mutable_data` method is "
         "deprecated since version 2.3, and `mutable_data` method "
         "will be removed in version 2.4! Please use "
         "`paddle::empty/full` method to create a new "
         "Tensor with allocated memory, and use data<T>() method "
         "to get the memory pointer of tensor instead. "
         "Reason: When calling `mutable_data` to allocate memory, "
         "the datatype, and data layout of tensor may be in "
         "an illegal state.";
  if (is_dense_tensor()) {
    return static_cast<phi::DenseTensor *>(impl_.get())->mutable_data<T>(place);
  }
  return nullptr;
}

template PADDLE_API bool *Tensor::mutable_data<bool>(const Place &place);
template PADDLE_API int8_t *Tensor::mutable_data<int8_t>(const Place &place);
template PADDLE_API uint8_t *Tensor::mutable_data<uint8_t>(const Place &place);
template PADDLE_API int16_t *Tensor::mutable_data<int16_t>(const Place &place);
template PADDLE_API int32_t *Tensor::mutable_data<int32_t>(const Place &place);
template PADDLE_API int64_t *Tensor::mutable_data<int64_t>(const Place &place);
template PADDLE_API phi::dtype::float16 *
Tensor::mutable_data<phi::dtype::float16>(const Place &place);
template PADDLE_API float *Tensor::mutable_data<float>(const Place &place);
template PADDLE_API double *Tensor::mutable_data<double>(const Place &place);
template PADDLE_API phi::dtype::complex<float>
    *Tensor::mutable_data<phi::dtype::complex<float>>(const Place &place);
template PADDLE_API phi::dtype::complex<double>
    *Tensor::mutable_data<phi::dtype::complex<double>>(const Place &place);

template <typename T>
const T *Tensor::data() const {
  if (is_dense_tensor()) {
    return static_cast<phi::DenseTensor *>(impl_.get())->data<T>();
  } else if (is_selected_rows()) {
    return static_cast<phi::SelectedRows *>(impl_.get())->value().data<T>();
  }
  return nullptr;
}

template PADDLE_API const bool *Tensor::data<bool>() const;
template PADDLE_API const int8_t *Tensor::data<int8_t>() const;
template PADDLE_API const uint8_t *Tensor::data<uint8_t>() const;
template PADDLE_API const int16_t *Tensor::data<int16_t>() const;
template PADDLE_API const uint16_t *Tensor::data<uint16_t>() const;
template PADDLE_API const int32_t *Tensor::data<int32_t>() const;
template PADDLE_API const uint32_t *Tensor::data<uint32_t>() const;
template PADDLE_API const int64_t *Tensor::data<int64_t>() const;
template PADDLE_API const uint64_t *Tensor::data<uint64_t>() const;
template PADDLE_API const phi::dtype::bfloat16 *
Tensor::data<phi::dtype::bfloat16>() const;
template PADDLE_API const phi::dtype::float16 *
Tensor::data<phi::dtype::float16>() const;
template PADDLE_API const float *Tensor::data<float>() const;
template PADDLE_API const double *Tensor::data<double>() const;
template PADDLE_API const phi::dtype::complex<float>
    *Tensor::data<phi::dtype::complex<float>>() const;
template PADDLE_API const phi::dtype::complex<double>
    *Tensor::data<phi::dtype::complex<double>>() const;
template PADDLE_API const phi::dtype::float8_e4m3fn *
Tensor::data<phi::dtype::float8_e4m3fn>() const;
template PADDLE_API const phi::dtype::float8_e5m2 *
Tensor::data<phi::dtype::float8_e5m2>() const;

template <typename T>
T *Tensor::data() {
  if (is_dense_tensor()) {
    return static_cast<phi::DenseTensor *>(impl_.get())->data<T>();
  } else if (is_selected_rows()) {
    return static_cast<phi::SelectedRows *>(impl_.get())
        ->mutable_value()
        ->data<T>();
  }
  return nullptr;
}

template PADDLE_API bool *Tensor::data<bool>();
template PADDLE_API int8_t *Tensor::data<int8_t>();
template PADDLE_API uint8_t *Tensor::data<uint8_t>();
template PADDLE_API int16_t *Tensor::data<int16_t>();
template PADDLE_API uint16_t *Tensor::data<uint16_t>();
template PADDLE_API int32_t *Tensor::data<int32_t>();
template PADDLE_API uint32_t *Tensor::data<uint32_t>();
template PADDLE_API int64_t *Tensor::data<int64_t>();
template PADDLE_API uint64_t *Tensor::data<uint64_t>();
template PADDLE_API phi::dtype::bfloat16 *Tensor::data<phi::dtype::bfloat16>();
template PADDLE_API phi::dtype::float16 *Tensor::data<phi::dtype::float16>();
template PADDLE_API float *Tensor::data<float>();
template PADDLE_API double *Tensor::data<double>();
template PADDLE_API phi::dtype::complex<float>
    *Tensor::data<phi::dtype::complex<float>>();
template PADDLE_API phi::dtype::complex<double>
    *Tensor::data<phi::dtype::complex<double>>();
template PADDLE_API phi::dtype::float8_e4m3fn *
Tensor::data<phi::dtype::float8_e4m3fn>();
template PADDLE_API phi::dtype::float8_e5m2 *
Tensor::data<phi::dtype::float8_e5m2>();

const void *Tensor::data() const {
  if (is_dense_tensor()) {
    return static_cast<phi::DenseTensor *>(impl_.get())->data();
  } else if (is_selected_rows()) {
    return static_cast<phi::SelectedRows *>(impl_.get())->value().data();
  }
  return nullptr;
}

void *Tensor::data() {
  if (is_dense_tensor()) {
    return static_cast<phi::DenseTensor *>(impl_.get())->data();
  } else if (is_selected_rows()) {
    return static_cast<phi::SelectedRows *>(impl_.get())
        ->mutable_value()
        ->data();
  }
  return nullptr;
}

// TODO(chenweihang): replace slice impl by API
Tensor Tensor::slice(int64_t begin_idx, int64_t end_idx) const {
  if (is_dense_tensor()) {
    return Tensor(
        std::make_shared<phi::DenseTensor>(phi::DenseTensorUtils::Slice(
            *(static_cast<phi::DenseTensor *>(impl_.get())),
            begin_idx,
            end_idx)));
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support slice operation on DenseTensor now."));
  }
}

const std::shared_ptr<phi::TensorBase> &Tensor::impl() const { return impl_; }

void Tensor::set_impl(const std::shared_ptr<phi::TensorBase> &impl) {
  impl_ = impl;
}

void Tensor::set_impl(std::shared_ptr<phi::TensorBase> &&impl) {
  impl_ = std::move(impl);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
gpuStream_t Tensor::stream() const {
  int device_id = phi::backends::gpu::GetCurrentDeviceId();
  auto *gpu_context = DeviceContextPool::Instance().Get<AllocationType::GPU>(
      GPUPlace(device_id));
  return gpu_context->stream();
}
#endif

const std::string &Tensor::name() const { return name_; }

void Tensor::set_name(const std::string &name) { name_ = name; }

/* Part 5: Status utils methods */

bool Tensor::defined() const { return impl_ != nullptr; }

bool Tensor::has_allocation() const {
  return defined() && impl_->has_allocation();
}

bool Tensor::initialized() const { return defined() && impl_->initialized(); }

bool Tensor::is_initialized() const {
  LOG_FIRST_N(WARNING, 1)
      << "The `is_initialized` method is deprecated since version "
         "2.3, and will be removed in version 2.4! "
         "Please use `initialized` method instead.";
  return defined() && impl_->initialized();
}

void Tensor::reset() {
  impl_.reset();
  autograd_meta_.reset();
  name_ = "";
}

/* Part 6: Operator overloading */

Tensor &Tensor::operator=(const Tensor &x) & = default;

Tensor &Tensor::operator=(Tensor &&x) &noexcept {
  impl_ = std::move(x.impl_);
  autograd_meta_ = std::move(x.autograd_meta_);
  name_ = std::move(x.name_);
  return *this;
}

AbstractAutogradMeta *Tensor::get_autograd_meta() const {
  return autograd_meta_.get();
}

const std::shared_ptr<AbstractAutogradMeta> &Tensor::mutable_autograd_meta()
    const {
  return autograd_meta_;
}

void Tensor::set_autograd_meta(
    std::shared_ptr<AbstractAutogradMeta> autograd_meta) {
  autograd_meta_ = std::move(autograd_meta);
}

void Tensor::bump_inplace_version() {
  if (is_dense_tensor()) {
    auto &inplace_version_counter =
        static_cast<phi::DenseTensor *>(impl_.get())->InplaceVersionCounter();
    inplace_version_counter.Bump();
  } else if (is_dist_tensor()) {
    auto &inplace_version_counter =
        static_cast<phi::distributed::DistTensor *>(impl_.get())
            ->unsafe_mutable_value()
            ->InplaceVersionCounter();
    inplace_version_counter.Bump();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "bump_inplace_version is only supported on "
        "DenseTensor and DistTensor now."));
  }
}

uint32_t Tensor::current_inplace_version() {
  if (is_dense_tensor()) {
    auto &inplace_version_counter =
        static_cast<phi::DenseTensor *>(impl_.get())->InplaceVersionCounter();
    return inplace_version_counter.CurrentVersion();
  } else if (is_dist_tensor()) {
    auto &inplace_version_counter =
        static_cast<phi::distributed::DistTensor *>(impl_.get())
            ->unsafe_mutable_value()
            ->InplaceVersionCounter();
    return inplace_version_counter.CurrentVersion();
  } else {
    LOG_FIRST_N(WARNING, 1) << "current_inplace_version is only supported on "
                               "DenseTensor DistTensor now.";
  }
  return 0;
}

void Tensor::reset_inplace_version(bool set_to_zero) {
  if (set_to_zero) {
    if (is_dense_tensor()) {
      auto &inplace_version_counter =
          static_cast<phi::DenseTensor *>(impl_.get())->InplaceVersionCounter();
      inplace_version_counter.SetInplaceVersionToZero();
    } else if (is_dist_tensor()) {
      auto &inplace_version_counter =
          static_cast<phi::distributed::DistTensor *>(impl_.get())
              ->unsafe_mutable_value()
              ->InplaceVersionCounter();
      return inplace_version_counter.SetInplaceVersionToZero();
    }
  }
}

/* Part 7: Contiguous methods */

bool Tensor::is_contiguous() const {
  if (is_dense_tensor() || is_dist_tensor()) {
    phi::DenseTensor *dense_tensor = nullptr;
    if (is_dist_tensor()) {
      dense_tensor = static_cast<phi::distributed::DistTensor *>(impl_.get())
                         ->unsafe_mutable_value();
    } else {
      dense_tensor = static_cast<phi::DenseTensor *>(impl_.get());
    }
    return dense_tensor->meta().is_contiguous();
  } else {
    PADDLE_THROW(
        common::errors::Unimplemented("Only support is_contiguous operation on "
                                      "DenseTensor or DistTensor now."));
  }
}

Tensor Tensor::contiguous() {
  if (is_dense_tensor() || is_dist_tensor()) {
    phi::DenseTensor *dense_tensor = nullptr;
    if (is_dist_tensor()) {
      dense_tensor = static_cast<phi::distributed::DistTensor *>(impl_.get())
                         ->unsafe_mutable_value();
    } else {
      dense_tensor = static_cast<phi::DenseTensor *>(impl_.get());
    }
    PADDLE_ENFORCE_NOT_NULL(dense_tensor,
                            common::errors::InvalidArgument(
                                "TensorImpl with nullptr is not supported"));
    if (!dense_tensor->meta().is_contiguous()) {
      auto new_dense_tensor = std::make_shared<phi::DenseTensor>();
      *new_dense_tensor = paddle::experimental::Trans2Contiguous(*dense_tensor);

      return Tensor(std::shared_ptr<phi::TensorBase>(new_dense_tensor),
                    autograd_meta_,
                    name_);
    } else {
      return *this;
    }
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support contiguous operation on DenseTensor or DistTensor now."));
  }
}

/* Part 13: Forward AD related*/
// The Forward AD API functions below are low level and are not to be used by
// end users who should use the API provided in torch/csrc/autograd.h

/// This function returns the forward gradient for this Tensor at the given
/// level.
const Tensor &Tensor::_fw_grad(uint64_t level) const {
  if (!autograd_meta_) return singleton_undefined_tensor;

  return autograd_meta_->fw_grad(level, *this);
}

// / This function can be used to set the value of the forward grad.
// / Note that the given new_grad might not be used directly if it has different
// / metadata (size/stride/storage offset) compared to this Tensor. In that
// case, / new_grad content will be copied into a new Tensor void
// _set_fw_grad(const paddle::Tensor& new_grad, uint64_t level, bool
// is_inplace_op) const {
//   // impl_->_set_fw_grad(*new_grad, *this, level, is_inplace_op);
// }

// const Tensor& Tensor::_fw_grad(
//     uint64_t level,
//     const TensorBase& self) const {
//   // See TensorImpl::grad() above for explanation about the line below
//   if (!autograd_meta_)
//     return Tensor();
//   return autograd_meta_->fw_grad(level, self);
// }

void Tensor::_set_fw_grad(const Tensor &new_grad,
                          uint64_t level,
                          bool is_inplace_op) {
  if (!autograd_meta_) {
    autograd_meta_ = std::make_shared<egr::AutogradMeta>();
  }
  autograd_meta_->set_fw_grad(new_grad, *this, level, is_inplace_op);
}

std::shared_ptr<phi::TensorBase> Tensor::get_impl() const {
  return this->impl_;
}

// Taken from codegened version
const Tensor &Tensor::_fw_primal(int64_t level) const {
  // auto& self_ = unpack(self, "self", 0);
  // std::shared_ptr<Identity> grad_fn;
  // if (compute_requires_grad( self )) {
  //   grad_fn = std::make_shared<Identity>();
  //   grad_fn->set_next_edges(collect_next_edges( self ));
  // }
  // auto tmp = ([&]() {
  //   at::AutoNonVariableTypeMode non_var_type_mode(true);
  //   return self_.alias();
  // })();
  // c10::optional<std::function<at::Tensor(const at::Tensor&)>>
  // func=c10::nullopt; if (!self.unsafeGetTensorImpl()->support_as_strided()) {
  //   auto size_vec = self.sizes().vec();
  //   func = [=](const at::Tensor& input_base) {
  //     return input_base.view(size_vec);
  //   };
  // }
  // auto result = as_view(/* base */ self, /* output */ tmp, /*
  // is_bw_differentiable */ true,
  //                       /* is_fw_differentiable */ false, /* view_func */
  //                       func, /* creation_meta */ CreationMeta::DEFAULT);
  // if (grad_fn) {
  //     set_history(flatten_tensor_args( result ), grad_fn);
  // }
  // if (generated::details::isFwGradDefined(self)) {
  //   // Modified from original codegen
  //   // We explicitly want to ignore the forward grad at the given level
  //   PD_CHECK(level == 0, "Invalid level given to _fw_primal");
  //   // End modified from original codegen
  // }
  // return result;
  return this->autograd_meta_->fw_grad(level, *this);
}

}  // namespace paddle
