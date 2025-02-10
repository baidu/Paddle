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
// disable numpy compile error

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <Python.h>
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/slice_utils.h"
#include "paddle/fluid/pybind/uva_utils.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/vocab/string_array.h"
#include "pybind11/detail/internals.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "paddle/common/ddim.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/nodes.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function_registry.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/memory/allocation/mmap_allocator.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/utils/pybind.h"

COMMON_DECLARE_bool(set_to_1d);
COMMON_DECLARE_bool(use_stride_kernel);

using egr::ConvertAllInputsToDistTensor;
using egr::InputsContainDistTensor;

namespace paddle::pybind {

extern void InitTensorWithNumpyValue(TensorObject* self,
                                     const pybind11::object& array,
                                     const phi::Place& place,
                                     bool zero_copy);

extern PyTypeObject* p_tensor_type;

Py_ssize_t GetSliceIndexFromPyObject(PyObject* obj) {
  if (PyObject_TypeCheck(obj, p_tensor_type)) {
    VLOG(6) << "Call GetSliceIndexFromTensor in Eager";
    paddle::Tensor tensor = CastPyArg2Tensor(obj, 0);
    PADDLE_ENFORCE_EQ(
        tensor.initialized(),
        true,
        common::errors::InvalidArgument(
            "We can only support initialized tensor in slice, however we got "
            "uninitialized tensor %s, please check your code.",
            tensor.name()));
    return GetSliceIndexFromTensor((*static_cast<phi::DenseTensor*>(
        CastPyArg2Tensor(obj, 0).impl().get())));
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "We should only get paddle::Tensor or VarBase in this "
        "method, when you reach this means we got another type index."));
  }
}

namespace {
#ifdef PADDLE_WITH_DISTRIBUTE
phi::DenseTensor ReshardXToReplicated(
    phi::distributed::DistTensor* dist_tensor) {
  if (!dist_tensor->dist_attr().is_replicated()) {
    phi::distributed::TensorDistAttr dist_attr(dist_tensor->dist_attr());
    std::vector<int64_t> dims_mapping(dist_tensor->dims().size(), -1);
    dist_attr.set_dims_mapping(dims_mapping);
    dist_attr.clean_partial_status();
    // reshard to replicate dist tensor
    VLOG(4) << "Reshard tensor: "
            << paddle::experimental::ReshardDebugInfo(*dist_tensor, dist_attr);
    auto* func =
        phi::distributed::ChooseProperReshardFunction(*dist_tensor, dist_attr);
    auto* dev_ctx =
        phi::DeviceContextPool::Instance().Get(dist_tensor->place());
    auto out_tensor = func->Eval(dev_ctx, *dist_tensor, dist_attr);
    return out_tensor->value();
  } else {
    return dist_tensor->value();
  }
}
#endif
}  // namespace

PyDoc_STRVAR(tensor_method_numpy__doc__,  // NOLINT
             R"DOC(numpy($self, /)
--

Returns a numpy array shows the value of current Tensor.

Returns:
    ndarray, The numpy value of current Tensor, dtype is
    same as current Tensor.

Examples:

    .. code-block:: python

        >>> import paddle
        >>> x = paddle.to_tensor([[1.0, 2.0, 3.0],
        ...                       [4.0, 5.0, 6.0]])
        >>> x.numpy()
        array([[1., 2., 3.],
               [4., 5., 6.]], dtype=float32)
)DOC");

static PyObject* tensor_method_numpy(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  auto& api = pybind11::detail::npy_api::get();
  if (!self->tensor.impl()) {
    Py_intptr_t py_dims[phi::DDim::kMaxRank];     // NOLINT
    Py_intptr_t py_strides[phi::DDim::kMaxRank];  // NOLINT
    py_dims[0] = 0;
    py_strides[0] = 0;

    PyObject* array = api.PyArray_NewFromDescr_(
        api.PyArray_Type_,
        api.PyArray_DescrFromType_(pybind11::detail::npy_api::NPY_FLOAT_),
        1,
        py_dims,
        py_strides,
        nullptr,
        pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
            pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
        nullptr);
    return array;
  }
  auto tensor_dims = self->tensor.shape();
  auto numpy_dtype = TensorDtype2NumpyDtype(self->tensor.type());
  auto sizeof_dtype = phi::SizeOf(self->tensor.type());
  Py_intptr_t py_dims[phi::DDim::kMaxRank];     // NOLINT
  Py_intptr_t py_strides[phi::DDim::kMaxRank];  // NOLINT
  size_t py_rank = tensor_dims.size();
  size_t numel = 1;
  if (self->tensor.is_dense_tensor()) {
    auto tensor_stride = self->tensor.strides();

    for (int i = static_cast<int>(tensor_dims.size()) - 1; i >= 0; --i) {
      py_dims[i] = static_cast<Py_intptr_t>(tensor_dims[i]);
      py_strides[i] = static_cast<Py_intptr_t>(sizeof_dtype * tensor_stride[i]);
      numel *= py_dims[i];
    }
  } else {
    for (int i = static_cast<int>(tensor_dims.size()) - 1; i >= 0; --i) {
      py_dims[i] = static_cast<Py_intptr_t>(tensor_dims[i]);
      py_strides[i] = static_cast<Py_intptr_t>(sizeof_dtype * numel);
      numel *= py_dims[i];
    }
  }

  if (!self->tensor.impl()->initialized()) {
    if (tensor_dims.empty()) {
      py_dims[0] = 0;
      py_strides[0] = 0;
      PyObject* array = api.PyArray_NewFromDescr_(
          api.PyArray_Type_,
          api.PyArray_DescrFromType_(numpy_dtype),
          1,
          py_dims,
          py_strides,
          nullptr,
          pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
              pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
          nullptr);
      return array;
    }

    // NOTE(zhiqiu): numpy will allocate memory automatically
    // if product of dims is not 0 and data is nullptr.
    // However, paddle's tensor with empty allocation means
    // not initialized. It is not consistent if tensor.numpy()
    // holds memory when tensor's allocation is empty.
    // so we emplace back a 0 to the dims to make it 0-size tensor.
    // For example, tensor with shape [2,3] becomes [2,3,0].
    auto contains_zero = false;
    for (size_t i = 0; i < py_rank; ++i) {
      if (py_dims[i] == 0) {
        contains_zero = true;
        break;
      }
    }
    if (!contains_zero) {
      py_dims[tensor_dims.size()] = 0;
      py_rank += 1;
      for (size_t i = 0; i < py_rank; ++i) {
        py_strides[i] = 0;
      }
    }

    PyObject* array = api.PyArray_NewFromDescr_(
        api.PyArray_Type_,
        api.PyArray_DescrFromType_(numpy_dtype),
        static_cast<int>(py_rank),
        py_dims,
        py_strides,
        nullptr,
        pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
            pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
        nullptr);
    return array;
  }

  phi::DenseTensor cpu_tensor;
  phi::CPUPlace cpu_place;

  if (self->tensor.is_cpu() || self->tensor.is_gpu_pinned()) {
    eager_gil_scoped_release guard;
    phi::CPUPlace place;
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor =
          static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      // deep copy
      paddle::memory::Copy(place,
                           cpu_tensor.Holder()->ptr(),
                           place,
                           dense_tensor->Holder()->ptr(),
                           dense_tensor->Holder()->size());
    } else if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
      VLOG(6) << "Getting DistTensor's numpy value";
      auto* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
      auto dense_tensor = ReshardXToReplicated(dist_tensor);

      cpu_tensor.set_meta(dense_tensor.meta());
      // deep copy
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor.Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      // deep copy
      paddle::memory::Copy(place,
                           cpu_tensor.Holder()->ptr(),
                           place,
                           dense_tensor.Holder()->ptr(),
                           dense_tensor.Holder()->size());
#else
      PADDLE_THROW(
          common::errors::Unavailable("The `numpy()` method of (Dist)Tensor "
                                      "is not supported in the current "
                                      "PaddlePaddle, please recompile and "
                                      "installPaddlePaddle with the option "
                                      "of `WITH_DISTRIBUTE=ON`."));
#endif
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      // deep copy
      paddle::memory::Copy(place,
                           cpu_tensor.Holder()->ptr(),
                           place,
                           dense_tensor->Holder()->ptr(),
                           dense_tensor->Holder()->size());
    }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (self->tensor.is_gpu()) {
    eager_gil_scoped_release guard;
#if defined(PADDLE_WITH_CUDA)
    gpuMemcpyKind kind = cudaMemcpyDeviceToHost;
#elif defined(PADDLE_WITH_HIP)
    gpuMemcpyKind kind = hipMemcpyDeviceToHost;
    phi::DeviceContextPool::Instance().Get(self->tensor.place())->Wait();
#endif
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor =
          static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      paddle::platform::GpuMemcpySync(cpu_tensor.Holder()->ptr(),
                                      dense_tensor->Holder()->ptr(),
                                      dense_tensor->Holder()->size(),
                                      kind);
    } else if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
      VLOG(6) << "Getting DistTensor's numpy value";
      auto* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
      auto dense_tensor = ReshardXToReplicated(dist_tensor);

      cpu_tensor.set_meta(dense_tensor.meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor.Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      paddle::platform::GpuMemcpySync(cpu_tensor.Holder()->ptr(),
                                      dense_tensor.Holder()->ptr(),
                                      dense_tensor.Holder()->size(),
                                      kind);
#else
      PADDLE_THROW(
          common::errors::Unavailable("The `numpy()` method of (Dist)Tensor "
                                      "is not supported in the current "
                                      "PaddlePaddle, please recompile and "
                                      "installPaddlePaddle with the option "
                                      "of `WITH_DISTRIBUTE=ON`."));
#endif
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      paddle::platform::GpuMemcpySync(cpu_tensor.Holder()->ptr(),
                                      dense_tensor->Holder()->ptr(),
                                      dense_tensor->Holder()->size(),
                                      kind);
    }
#endif
#if defined(PADDLE_WITH_XPU)
  } else if (self->tensor.is_xpu()) {
    phi::CPUPlace place;
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor =
          static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      paddle::memory::Copy(place,
                           cpu_tensor.Holder()->ptr(),
                           dense_tensor->place(),
                           dense_tensor->Holder()->ptr(),
                           dense_tensor->Holder()->size());
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      paddle::memory::Copy(place,
                           cpu_tensor.Holder()->ptr(),
                           dense_tensor->place(),
                           dense_tensor->Holder()->ptr(),
                           dense_tensor->Holder()->size());
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (self->tensor.is_custom_device()) {
    eager_gil_scoped_release guard;
    phi::DeviceContextPool::Instance().Get(self->tensor.place())->Wait();
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor =
          static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      phi::DeviceManager::GetDeviceWithPlace(self->tensor.place())
          ->MemoryCopyD2H(cpu_tensor.Holder()->ptr(),
                          dense_tensor->Holder()->ptr(),
                          dense_tensor->Holder()->size());
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      // TODO(qili93): temporary for ascend npu performance to be removed along
      // with npu_identity op
      paddle::Tensor temp_tensor(std::make_shared<phi::DenseTensor>());
      if (dense_tensor->storage_properties_initialized()) {
        temp_tensor = npu_identity_ad_func(self->tensor, -1);
        dense_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(temp_tensor.impl());
      }
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      phi::DeviceManager::GetDeviceWithPlace(self->tensor.place())
          ->MemoryCopyD2H(cpu_tensor.Holder()->ptr(),
                          dense_tensor->Holder()->ptr(),
                          dense_tensor->Holder()->size());
    }
#endif
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Tensor.numpy() only support cpu tensor."));
    RETURN_PY_NONE
  }

  void* array_buffer = cpu_tensor.Holder()->ptr();
  size_t array_offset = cpu_tensor.offset();

  PyObject* base = ToPyObject(paddle::Tensor(
      std::make_shared<phi::DenseTensor>(std::move(cpu_tensor))));
  uintptr_t ptr = reinterpret_cast<uintptr_t>(array_buffer) + array_offset;
  PyObject* array = api.PyArray_NewFromDescr_(
      api.PyArray_Type_,
      api.PyArray_DescrFromType_(numpy_dtype),
      static_cast<int>(py_rank),
      py_dims,
      py_strides,
      reinterpret_cast<void*>(ptr),
      pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
          pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
      nullptr);

  api.PyArray_SetBaseObject_(array, base);

  return array;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_numpy_for_string_tensor(TensorObject* self,
                                                       PyObject* args,
                                                       PyObject* kwargs) {
  EAGER_TRY
  auto& api = pybind11::detail::npy_api::get();
  if (!self->tensor.impl() || !self->tensor.impl()->initialized()) {
    VLOG(6) << "The StringTensor is uninitialized. Return the empty string "
               "numpy array.";
    Py_intptr_t py_dims[phi::DDim::kMaxRank];     // NOLINT
    Py_intptr_t py_strides[phi::DDim::kMaxRank];  // NOLINT
    py_dims[0] = 0;
    py_strides[0] = 0;

    PyObject* array = api.PyArray_NewFromDescr_(
        api.PyArray_Type_,
        api.PyArray_DescrFromType_(pybind11::detail::npy_api::NPY_UNICODE_),
        1,
        py_dims,
        py_strides,
        nullptr,
        pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
            pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
        nullptr);
    return array;
  }

  if (self->tensor.is_cpu()) {
    VLOG(6) << "Getting StringTensor's numpy value";
    auto string_tensor =
        std::dynamic_pointer_cast<phi::StringTensor>(self->tensor.impl());
    const auto* st_ptr = string_tensor->data();
    auto numel = self->tensor.numel();
    auto tensor_dims = self->tensor.shape();
    // Get the max unicode length of StringTensor to create numpy unicode
    // string array.
    auto* longest_pstring = std::max_element(
        st_ptr, st_ptr + numel, [](const auto& a, const auto& b) {
          auto a_unicode_len =
              phi::strings::GetUnicodeStrLen(a.data(), a.size());
          auto b_unicode_len =
              phi::strings::GetUnicodeStrLen(b.data(), b.size());
          return a_unicode_len < b_unicode_len;
        });
    size_t max_unicode_length = phi::strings::GetUnicodeStrLen(
        longest_pstring->data(), longest_pstring->size());
    max_unicode_length = (max_unicode_length == 0) ? 1 : max_unicode_length;
    VLOG(6) << "The max unicode length is " << max_unicode_length;
    auto sp =
        std::make_unique<uint32_t[]>(max_unicode_length * numel);  // NOLINT
    auto py_array_data = sp.get();
    memset(py_array_data, 0, max_unicode_length * numel * sizeof(uint32_t));
    for (int64_t i = 0; i < numel; ++i) {
      auto curr_unicode_len =
          phi::strings::GetUnicodeStrLen(st_ptr[i].data(), st_ptr[i].size());
      phi::strings::GetUnicodeStr(st_ptr[i].data(),
                                  py_array_data + i * max_unicode_length,
                                  curr_unicode_len);
    }
    py::array array(py::dtype("U" + std::to_string(max_unicode_length)),
                    tensor_dims,
                    {},
                    py_array_data);
    return array.release().ptr();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "StringTensor.numpy() only support cpu tensor."));
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__is_initialized(TensorObject* self,
                                               PyObject* args,
                                               PyObject* kwargs) {
  EAGER_TRY
  return ToPyObject(self->tensor.initialized());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__is_dense_tensor_hold_allocation(
    TensorObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  if (self->tensor.is_dense_tensor()) {
    auto dense_tensor_ptr =
        std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
    return ToPyObject(
        dense_tensor_ptr->IsInitialized() &&
        ((dense_tensor_ptr->numel() > 0 && dense_tensor_ptr->Holder()->ptr()) ||
         dense_tensor_ptr->numel() == 0));
  } else if (self->tensor.is_dist_tensor()) {
    auto dense_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get())
            ->value();
    return ToPyObject(
        dense_tensor.IsInitialized() &&
        ((dense_tensor.numel() > 0 && dense_tensor.Holder()->ptr()) ||
         dense_tensor.numel() == 0));
  } else {
    return ToPyObject(false);
  }

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static void IncreaseTensorReferenceCountUntilCopyComplete(
    const paddle::Tensor& tensor, const phi::Place& place) {
  auto place_ = phi::is_gpu_place(place) ? place : tensor.place();

  auto tracer = egr::Controller::Instance().GetCurrentTracer();
  auto gc = tracer->MutableGarbageCollectorIfNotExists(place_);

  // Note(dev): This is an empty callback, the only way is to "reference"
  // inner memory Holder, so it will not be destructed until the kernels
  // launched at current stream of given place is finished, such as
  // CUDAPinned Mem -> CUDA by cudaMemcpyAsync.
  auto callback = [tensor, place_]() {
    VLOG(3) << "Run callback of Tensor:" << tensor.name() << " at place "
            << place_;
  };
  gc->DirectClearCallback(callback);
}

static PyObject* tensor_method__copy_to(TensorObject* self,
                                        PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 0), 0);
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 1), 1);
  paddle::Tensor cp_tensor;
  {
    eager_gil_scoped_release guard;

    EagerSetDeviceId();

    cp_tensor = self->tensor.copy_to(place, blocking);
    if (!blocking) {
      IncreaseTensorReferenceCountUntilCopyComplete(self->tensor, place);
    }
    egr::EagerUtils::autograd_meta(&cp_tensor)->SetStopGradient(true);
    egr::EagerUtils::autograd_meta(&cp_tensor)
        ->SetPersistable(
            egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  }
  return ToPyObject(cp_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_reconstruct_from___doc__,  // NOLINT
             R"DOC(reconstruct_from_($self, other/)
--

Reconstruct the self with other Tensor. It is a deep copy of 'self = other'.

Returns:
    None.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> t1 = paddle.to_tensor([1.0], stop_gradient=False)
        >>> t2 = paddle.to_tensor([2.0], stop_gradient=True)

        >>> t1.reconstruct_from_(t2)
        >>> print(t1)
        Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [2.])
)DOC");

static PyObject* tensor_method_reconstruct_from_(TensorObject* self,
                                                 PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor src_tensor = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  std::string orig_name = self->tensor.name();
  VLOG(6) << "Start Reconstructing Tensor from" << src_tensor.name() << " to "
          << orig_name;
  self->tensor = src_tensor;

  // Recover source name
  self->tensor.set_name(orig_name);

  VLOG(6) << "Finished Reconstructing Tensor from" << src_tensor.name()
          << " to " << self->tensor.name();
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_copy_(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor& src_tensor = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  const phi::distributed::ProcessMesh* mesh = nullptr;
  if (InputsContainDistTensor(&mesh, src_tensor, self->tensor)) {
    ConvertAllInputsToDistTensor(mesh, src_tensor, self->tensor);
  }

  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 1), 1);
  VLOG(6) << "Start Copy Tensor " << src_tensor.name() << " to "
          << self->tensor.name();
  if (!self->tensor.initialized()) {
    eager_gil_scoped_release guard;

    EagerSetDeviceId();

    egr::EagerUtils::autograd_meta(&(self->tensor))
        ->SetStopGradient(
            egr::EagerUtils::autograd_meta(&(src_tensor))->StopGradient());
    egr::EagerUtils::autograd_meta(&(self->tensor))
        ->SetPersistable(
            egr::EagerUtils::autograd_meta(&(src_tensor))->Persistable());
    if (src_tensor.has_allocation()) {
      self->tensor.copy_(src_tensor, src_tensor.place(), blocking);
    }
  } else {
    if (src_tensor.has_allocation()) {
      eager_gil_scoped_release guard;

      EagerSetDeviceId();

      self->tensor.copy_(src_tensor, self->tensor.place(), blocking);
    }
  }

  VLOG(6) << "Finish Copy Tensor " << src_tensor.name() << " to "
          << self->tensor.name();
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_clone__doc__,  // NOLINT
             R"DOC(clone($self, /)
--

Returns a new Tensor, which is clone of origin Tensor, and it remains in the current graph.
It will always have a Tensor copy.
In addition, the cloned Tensor provides gradient propagation.

Returns:
    Tensor, The cloned Tensor.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor(1.0, stop_gradient=False)
        >>> clone_x = x.clone()
        >>> clone_x.retain_grads()
        >>> y = clone_x**2
        >>> y.backward()
        >>> print(clone_x.stop_gradient)
        False
        >>> print(clone_x.grad)
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False, 2.)
        >>> print(x.stop_gradient)
        False
        >>> print(x.grad)
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False, 2.)

        >>> x = paddle.to_tensor(1.0)
        >>> clone_x = x.clone()
        >>> clone_x.stop_gradient = False
        >>> z = clone_x**3
        >>> z.backward()
        >>> print(clone_x.stop_gradient)
        False
        >>> print(clone_x.grad)
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False, 3.)
        >>> print(x.stop_gradient)
        True
        >>> print(x.grad)
        None
)DOC");

static PyObject* tensor_method_clone(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor out;
  {
    eager_gil_scoped_release guard;

    EagerSetDeviceId();

    PADDLE_ENFORCE_EQ(
        self->tensor.has_allocation(),
        true,
        common::errors::InvalidArgument(
            "We can only support initialized tensor in clone, however we got "
            "uninitialized tensor %s, please check your code.",
            self->tensor.name()));

    out = assign_ad_func(self->tensor);
  }
  return ToPyObject(out);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_retain_grads__doc__, R"DOC(retain_grads($self, /)
--

Enables this Tensor to have their grad populated during backward(). It is a no-op for leaf tensors.

Returns:
    None.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1.0, 2.0, 3.0])
        >>> x.stop_gradient = False
        >>> y = x + x
        >>> y.retain_grads()
        >>> loss = y.sum()
        >>> loss.backward()

        >>> print(y.grad)
        Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=False,
        [1., 1., 1.])

        >>> x = paddle.to_tensor([1.0, 2.0, 3.0])
        >>> x.stop_gradient = False
        >>> y = x + x
        >>> y.retain_grads()
        >>> loss = y.sum()
        >>> loss.backward()

        >>> print(y.grad)
        Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=False,
        [1., 1., 1.])
)DOC");  // NOLINT

static PyObject* tensor_retain_grads(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  if (egr::Controller::Instance().HasGrad()) {
    eager_gil_scoped_release guard;
    auto meta = egr::EagerUtils::autograd_meta(&(self->tensor));
    if (!meta->GetMutableGradNode()) {
      VLOG(6) << "Make grad node of tensor: " << self->tensor.name()
              << "become accumulation node";
      meta->SetGradNode(std::make_shared<egr::GradNodeAccumulation>(meta));
    }
    egr::egr_utils_api::RetainGradForTensor(self->tensor);
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_clear_gradient__doc__,  // NOLINT
             R"DOC(clear_gradient($self, set_to_zero=True, /)
--

Only for Tensor that has gradient, normally we use this for Parameters since
other temporary Tensor doesn't has gradient.

The Gradient of current Tensor will be set to ``0`` elementwise or ``None``.

Args:
    set_to_zero (bool, optional): If set to ``True``, the gradient will be set
        to ``0`` elementwise, otherwise the gradient will be set to ``None``.
        Default: ``True``.

Returns:
    None.

Examples:

    .. code-block:: python

        >>> import paddle
        >>> input = paddle.uniform([10, 2])
        >>> linear = paddle.nn.Linear(2, 3)
        >>> out = linear(input)
        >>> out.backward()
        >>> print("Before clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))
        >>> # doctest: +SKIP("Random output")
        Before clear_gradient, linear.weight.grad: Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=False,
        [[-0.03178465, -0.03178465, -0.03178465],
         [-0.98546225, -0.98546225, -0.98546225]])
        >>> # doctest: -SKIP
        >>> linear.weight.clear_gradient()
        >>> print("After clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))
        After clear_gradient, linear.weight.grad: Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=False,
        [[0., 0., 0.],
         [0., 0., 0.]])

)DOC");

static PyObject* tensor_clear_gradient(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "ClearGradient " << self->tensor.name();

  Py_ssize_t args_num = PyTuple_Size(args);
  bool set_to_zero = true;
  if (args_num == (Py_ssize_t)1) {
    set_to_zero = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 0), 0);
  }

  paddle::Tensor* grad = nullptr;
  bool is_leaf = egr::EagerUtils::IsLeafTensor(self->tensor);
  if (is_leaf) {
    grad = egr::EagerUtils::mutable_grad(self->tensor);
    PADDLE_ENFORCE(
        grad != nullptr,
        common::errors::Fatal("Detected nullptr grad"
                              "Please check if you have manually cleared"
                              "the grad inside autograd_meta"));
  } else {
    auto meta = egr::EagerUtils::unsafe_autograd_meta(self->tensor);
    grad = meta->MutableGrad();
  }

  if (grad->impl()) {
    eager_gil_scoped_release guard;
    if (grad->is_selected_rows()) {
      auto selected_rows =
          std::dynamic_pointer_cast<phi::SelectedRows>(grad->impl());
      if (selected_rows->mutable_value()->IsInitialized()) {
        selected_rows->mutable_rows()->clear();
        selected_rows->mutable_value()->clear();
      }
    } else if (grad->is_dense_tensor() || grad->is_dist_tensor()) {
      if (grad->initialized()) {
        phi::DenseTensor* grad_t = nullptr;
        if (grad->is_dense_tensor()) {
          grad_t = static_cast<phi::DenseTensor*>(grad->impl().get());
        } else {
          grad_t =
              static_cast<phi::distributed::DistTensor*>(grad->impl().get())
                  ->unsafe_mutable_value();
        }
        if (set_to_zero) {
          EagerSetDeviceId();
          auto* dev_ctx =
              phi::DeviceContextPool::Instance().Get(grad_t->place());
          phi::funcs::set_constant(*dev_ctx, grad_t, 0.0);
          if (is_leaf) {
            std::static_pointer_cast<egr::GradNodeAccumulation>(
                egr::EagerUtils::grad_node(self->tensor))
                ->SetFakeEmpty(true);
          }
        } else {
          VLOG(4) << "Gradient of " << self->tensor.name()
                  << " is initialized, will be released.";
          grad_t->MoveMemoryHolder();
        }
      }
    }
  }

  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__zero_grads(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "ZeroGrads " << self->tensor.name();

  if (egr::EagerUtils::IsLeafTensor(self->tensor)) {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    // Add RetainGrad as PostHook to AccumulationNode
    paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
    PADDLE_ENFORCE(
        grad != nullptr,
        common::errors::Fatal("Detected nullptr grad"
                              "Please check if you have manually cleared"
                              "the grad inside autograd_meta"));
    if (grad->initialized()) {
      if (grad->is_dense_tensor() || grad->is_dist_tensor()) {
        phi::DenseTensor* t = nullptr;
        if (grad->is_dense_tensor()) {
          t = static_cast<phi::DenseTensor*>(grad->impl().get());
        } else {
          t = static_cast<phi::distributed::DistTensor*>(grad->impl().get())
                  ->unsafe_mutable_value();
        }
        auto* dev_ctx = phi::DeviceContextPool::Instance().Get(t->place());
        phi::funcs::set_constant(*dev_ctx, t, 0.0);
      } else {
        grad->set_impl(paddle::experimental::zeros_like(*(grad)).impl());
      }
    }
  } else {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    auto meta = egr::EagerUtils::unsafe_autograd_meta(self->tensor);
    if (meta->MutableGrad()->initialized()) {
      if (meta->MutableGrad()->is_dense_tensor() ||
          meta->MutableGrad()->is_dist_tensor()) {
        phi::DenseTensor* t = nullptr;
        if (meta->MutableGrad()->is_dense_tensor()) {
          t = static_cast<phi::DenseTensor*>(meta->MutableGrad()->impl().get());
        } else {
          t = static_cast<phi::distributed::DistTensor*>(
                  meta->MutableGrad()->impl().get())
                  ->unsafe_mutable_value();
        }
        auto* dev_ctx = phi::DeviceContextPool::Instance().Get(t->place());
        phi::funcs::set_constant(*dev_ctx, t, 0.0);
      } else {
        meta->MutableGrad()->set_impl(
            paddle::experimental::zeros_like(*(meta->MutableGrad())).impl());
      }
    }
  }

  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__to_dist(TensorObject* self,
                                 PyObject* args,
                                 PyObject* kwargs) {
  EAGER_TRY
  const auto& placements =
      CastPyArg2VectorOfPlacement(PyTuple_GET_ITEM(args, 0), 0);
  const auto& mesh = CastPyArg2ProcessMesh(PyTuple_GET_ITEM(args, 1), 1);

  if (self->tensor.is_dense_tensor()) {
    const auto& dense_tensor_ptr =
        std::static_pointer_cast<phi::DenseTensor>(self->tensor.impl());
    auto dist_tensor_ptr = std::make_shared<phi::distributed::DistTensor>(
        dense_tensor_ptr, mesh, placements);
    self->tensor.set_impl(dist_tensor_ptr);
  }

  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__share_buffer_to(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* dst_ptr =
      &(reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor);
  if (!self->tensor.initialized()) {
    if (self->tensor.numel() == 0) {
      // Do nothing for 0-size Tensor
      Py_RETURN_NONE;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Tensor %s has not been initialized! please initialize "
          "src tensor before share_buffer_with to other.",
          self->tensor.name()));
    }
  }
  if (self->tensor.is_dist_tensor()) {
    auto* src_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get())
            ->unsafe_mutable_value();
    if (!src_tensor->meta().is_contiguous()) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Tensor %s is not contiguous, cannot call share_buffer_to on it.",
          self->tensor.name()));
    }
    if (!dst_ptr->defined()) {
      dst_ptr->set_impl(std::make_shared<phi::distributed::DistTensor>());
    }
    auto dst_tensor =
        static_cast<phi::distributed::DistTensor*>(dst_ptr->impl().get())
            ->unsafe_mutable_value();
    dst_tensor->ShareBufferWith(*src_tensor);
    dst_tensor->ShareDataTypeWith(*src_tensor);
  } else {
    auto* src_tensor =
        static_cast<phi::DenseTensor*>(self->tensor.impl().get());
    if (!src_tensor->meta().is_contiguous()) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Tensor %s is not contiguous, cannot call share_buffer_to on it.",
          self->tensor.name()));
    }
    if (!dst_ptr->defined()) {
      dst_ptr->set_impl(std::make_shared<phi::DenseTensor>());
    }
    auto dst_tensor = static_cast<phi::DenseTensor*>(dst_ptr->impl().get());
    dst_tensor->ShareBufferWith(*src_tensor);
    dst_tensor->ShareDataTypeWith(*src_tensor);
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__unsafe_share_buffer_to(TensorObject* self,
                                                PyObject* args,
                                                PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* dst_ptr =
      &(reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor);
  if (!self->tensor.initialized()) {
    if (self->tensor.numel() == 0) {
      // Do nothing for 0-size Tensor
      Py_RETURN_NONE;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Tensor %s has not been initialized! please initialize "
          "src tensor before share_buffer_with to other.",
          self->tensor.name()));
    }
  }
  if (self->tensor.is_dist_tensor()) {
    auto* src_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get())
            ->unsafe_mutable_value();
    if (!dst_ptr->defined()) {
      dst_ptr->set_impl(std::make_shared<phi::distributed::DistTensor>());
    }
    auto dst_tensor =
        static_cast<phi::distributed::DistTensor*>(dst_ptr->impl().get())
            ->unsafe_mutable_value();
    dst_tensor->ShareBufferWith(*src_tensor);
    dst_tensor->ShareDataTypeWith(*src_tensor);
  } else {
    auto* src_tensor =
        static_cast<phi::DenseTensor*>(self->tensor.impl().get());
    if (!dst_ptr->defined()) {
      dst_ptr->set_impl(std::make_shared<phi::DenseTensor>());
    }
    auto dst_tensor = static_cast<phi::DenseTensor*>(dst_ptr->impl().get());
    dst_tensor->ShareBufferWith(*src_tensor);
    dst_tensor->ShareDataTypeWith(*src_tensor);
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__is_shared_buffer_with(TensorObject* self,
                                               PyObject* args,
                                               PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* dst_ptr =
      &(reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor);
  PADDLE_ENFORCE_EQ(self->tensor.has_allocation(),
                    true,
                    common::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        self->tensor.name()));
  bool res = false;
  if (!self->tensor.defined() || !dst_ptr->defined()) {
    return ToPyObject(res);
  }
  if (self->tensor.is_dist_tensor()) {
    auto* self_ptr =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get())
            ->unsafe_mutable_value();
    auto dst_tensor =
        static_cast<phi::distributed::DistTensor*>(dst_ptr->impl().get())
            ->unsafe_mutable_value();
    res = dst_tensor->IsSharedBufferWith(*self_ptr);
    return ToPyObject(res);
  } else {
    auto* self_ptr = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
    auto dst_tensor = static_cast<phi::DenseTensor*>(dst_ptr->impl().get());
    res = dst_tensor->IsSharedBufferWith(*self_ptr);
    return ToPyObject(res);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__share_underline_tensor_to(TensorObject* self,
                                                   PyObject* args,
                                                   PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* src_ptr =
      &(reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor);
  if (!self->tensor.has_allocation()) {
    PADDLE_ENFORCE(
        self->tensor.is_dist_tensor() &&
            !phi::distributed::IsCurRankInMesh(
                static_cast<phi::distributed::DistTensor*>(
                    self->tensor.impl().get())
                    ->process_mesh()),
        common::errors::InvalidArgument(
            "Tensor %s either lacks impl_ or holder_, Please initialize "
            "src tensor before share_buffer_with to other.",
            self->tensor.name()));
  }
  src_ptr->set_impl(self->tensor.impl());
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__is_shared_underline_tensor_with(TensorObject* self,
                                                         PyObject* args,
                                                         PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor src_tensor = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  PADDLE_ENFORCE_EQ(self->tensor.has_allocation(),
                    true,
                    common::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        src_tensor.name()));
  bool res = false;
  if (!self->tensor.defined() || !src_tensor.defined()) {
    return ToPyObject(res);
  }
  res = (self->tensor.impl().get() == src_tensor.impl().get());
  return ToPyObject(res);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_detach__doc__,  // NOLINT
             R"DOC(detach($self, /)
--

Returns a new Tensor, detached from the current graph.
It will share data with origin Tensor and always doesn't have a Tensor copy.
In addition, the detached Tensor doesn't provide gradient propagation.

Returns:
    Tensor, The detached Tensor.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1.0], stop_gradient=False)
        >>> detach_x = x.detach()
        >>> detach_x[0] = 10.0
        >>> print(x)
        Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=False, [10.])

        >>> y = x**2
        >>> y.backward()
        >>> print(x.grad)
        Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=False, [20.])

        >>> print(detach_x.grad) # None, 'stop_gradient=True' by default
        None

        >>> detach_x.stop_gradient = False # Set stop_gradient to be False, supported auto-grad
        >>> z = detach_x**3
        >>> z.backward()

        >>> print(x.grad)
        Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=False, [20.])

        >>> print(detach_x.grad)
        Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=False, [300.])

        >>> # Due to sharing of data with origin Tensor, There are some unsafe operations:
        >>> # y = 2 * x
        >>> # detach_x[:] = 5.0
        >>> # y.backward()
        >>> # It will raise Error:
        >>> #   one of the variables needed for gradient computation has been modified by an inplace operation.
)DOC");

static PyObject* tensor_method_detach(TensorObject* self,
                                      PyObject* args,
                                      PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE_EQ(
      self->tensor.defined(),
      true,
      common::errors::InvalidArgument("Tensor %s has not been initialized!",
                                      self->tensor.name()));

  PyObject* obj = p_tensor_type->tp_alloc(p_tensor_type, 0);
  if (obj) {
    auto v = reinterpret_cast<TensorObject*>(obj);
    new (&(v->tensor)) paddle::Tensor();
    v->tensor.set_impl(self->tensor.impl());
    v->tensor.set_name(egr::Controller::Instance().GenerateUniqueName());
    auto autograd_meta_src = egr::EagerUtils::autograd_meta(&(self->tensor));
    auto autograd_meta = egr::EagerUtils::autograd_meta(&(v->tensor));
    autograd_meta->SetPersistable(autograd_meta_src->Persistable());
  } else {
    PADDLE_THROW(
        common::errors::Fatal("tp_alloc return null, can not new a PyObject."));
  }

  return obj;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_detach___doc__, R"DOC(detach_($self, /)
--

Detach self from the current graph, and returns self Tensor.
In addition, the detached Tensor doesn't provide gradient propagation.

Returns:
    Tensor, The detached Tensor.
)DOC");

static PyObject* tensor_method_detach_(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE_EQ(
      self->tensor.defined(),
      true,
      common::errors::InvalidArgument("Tensor %s has not been initialized!",
                                      self->tensor.name()));

  auto autograd_meta = std::make_shared<egr::AutogradMeta>();
  autograd_meta->SetPersistable(
      egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  self->tensor.set_autograd_meta(autograd_meta);
  Py_INCREF(reinterpret_cast<PyObject*>(self));
  return reinterpret_cast<PyObject*>(self);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}  // NOLINT

PyDoc_STRVAR(tensor_method_get_tensor__doc__, R"DOC(get_tensor($self, /)
--

Returns the underline tensor in the origin Tensor.

Returns:
    Underline tensor.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1.0], stop_gradient=False)
        >>> underline_x = x.get_tensor()
        >>> print(underline_x)
          - place: Place(cpu)
          - shape: [1]
          - layout: NCHW
          - dtype: float32
          - data: [1]
)DOC");  // NOLINT

static PyObject* tensor_method_get_underline_tensor(TensorObject* self,
                                                    PyObject* args,
                                                    PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    // The original `get_tensor` method of Variable will create a empty tensor
    phi::DenseTensor empty_tensor;
    return ToPyObject(&empty_tensor);
  }
  if (self->tensor.is_dense_tensor()) {
    auto* tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
    VLOG(6) << "tensor: " << tensor->IsInitialized();
    return ToPyObject(tensor);
  } else if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
    auto* tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
    VLOG(6) << "dist tensor: " << tensor->defined();
    return ToPyObject(tensor);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "The `get_tensor()` method of (Dist)Tensor is not supported in the "
        "current PaddlePaddle, please recompile and installPaddlePaddle "
        "with the option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_get_underline_selected_rows(TensorObject* self,
                                                           PyObject* args,
                                                           PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    RETURN_PY_NONE
  }
  if (self->tensor.is_selected_rows()) {
    auto* selected_rows =
        static_cast<phi::SelectedRows*>(self->tensor.impl().get());
    return ToPyObject(selected_rows);
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__get_tensor_from_selected_rows(
    TensorObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(
      self->tensor.is_selected_rows(),
      common::errors::Fatal("this method is only effective for SelectedRows."));

  auto* selected_rows =
      static_cast<phi::SelectedRows*>(self->tensor.impl().get());

  PADDLE_ENFORCE(selected_rows->initialized(),
                 common::errors::Fatal("SelectedRows must be initialized."));

  auto* dense_tensor =
      static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
  VLOG(4) << "dense_tensor: " << dense_tensor->IsInitialized();

  auto t = paddle::Tensor(egr::Controller::Instance().GenerateUniqueName());
  t.set_impl(std::make_shared<phi::DenseTensor>(*dense_tensor));

  return ToPyObject(t);

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__getitem_dygraph(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  SetPythonStack();
  PyObject* _index = PyTuple_GET_ITEM(args, 0);
  VLOG(4) << "Call new indexing strategy _getitem_dygraph";

  // Note(0x45f): Using defined() instead of initialized()
  // to support slice tensor which shape like [0, 0, 0].
  PADDLE_ENFORCE_EQ(
      self->tensor.defined(),
      true,
      common::errors::InvalidArgument(
          "tensor %s has not been initialized, we can only slice initialized "
          "tensor please init it first with numpy or other tensor.",
          self->tensor.name()));

  auto tensor = self->tensor;
  const int rank = tensor.shape().size();
  std::vector<int> slice_starts, slice_ends, slice_strides;
  std::vector<int64_t> slice_axes, decrease_axis, infer_flags, none_axes;

  bool has_advanced_index = false;
  bool use_strided_slice = false;
  std::vector<int> advanced_index_dim(
      rank * 2,
      -1);  // content is dim, multiply 2 is to avoid all index are None
  std::vector<paddle::Tensor> advanced_index;  // content is index tensor

  // step1: parsing the index and recording them
  ParseIndex(tensor,
             _index,
             &slice_axes,
             &slice_starts,
             &slice_ends,
             &slice_strides,
             &decrease_axis,
             &none_axes,
             &infer_flags,
             &advanced_index_dim,
             &advanced_index,
             &has_advanced_index,
             &use_strided_slice);

  // step2: Dealing with basic indexing
  bool out_is_view = false;
  auto out = getTensorWithBasicIndexing(tensor,
                                        &slice_axes,
                                        &slice_starts,
                                        &slice_ends,
                                        &slice_strides,
                                        &decrease_axis,
                                        &none_axes,
                                        &infer_flags,
                                        &use_strided_slice,
                                        &out_is_view);

  if (!has_advanced_index) {
    return ToPyObject(out);
  }

  // step3: Dealing with advanced indexing
  std::vector<paddle::Tensor> transed_index;
  std::vector<int> trans_back_dim, trans_dim;
  int pos_of_new_dim = INT_MAX, rank_of_new_dim = 1;

  paddle::Tensor transed_tensor = dealWithAdvancedIndex(out,
                                                        &advanced_index_dim,
                                                        &advanced_index,
                                                        false,
                                                        &transed_index,
                                                        &trans_back_dim,
                                                        &pos_of_new_dim,
                                                        &rank_of_new_dim,
                                                        &trans_dim,
                                                        &out_is_view);

  if (transed_index.size() == 1 &&
      transed_index[0].dtype() == phi::DataType::BOOL) {
    // get value for bool tensor
    out = getValueForBoolTensor(transed_tensor, transed_index[0]);
  } else {
    // get value for int tensor
    ParseBoolAndBroadcastIndices(&transed_index);
    paddle::Tensor transed_advanced_index_tensor;
    if (transed_index.size() > 1) {
      transed_advanced_index_tensor = stack_ad_func(transed_index, -1);
    } else {
      // fast path for single index tensor, since stack is much slower than
      // unsqueeze
      transed_advanced_index_tensor = unsqueeze_ad_func(transed_index[0], {-1});
    }

    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(
            &mesh, transed_tensor, transed_advanced_index_tensor)) {
      ConvertAllInputsToDistTensor(
          mesh, transed_tensor, transed_advanced_index_tensor);
    }

    out = gather_nd_ad_func(transed_tensor, transed_advanced_index_tensor);
  }

  if (pos_of_new_dim != 0) {
    std::vector<int> perm(out.shape().size(), 0);
    int tmp1 = rank_of_new_dim, tmp2 = 0,
        tmp3 = pos_of_new_dim + rank_of_new_dim;
    for (int i = 0; i < static_cast<int>(out.shape().size()); ++i) {
      if (i < pos_of_new_dim) {
        perm[i] =
            tmp1++;  // range(rank_of_new_dim, pos_of_new_dim + rank_of_new_dim)
      } else if (i >= pos_of_new_dim && i < pos_of_new_dim + rank_of_new_dim) {
        perm[i] = tmp2++;  // range(0, rank_of_new_dim)
      } else {
        perm[i] = tmp3++;  // range(pos_of_new_dim + rank_of_new_dim, out.ndim)
      }
    }

    out = transpose_ad_func(out, perm);
  }

  return ToPyObject(out);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__getitem_from_offset(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  phi::DenseTensor* ptr = nullptr;
  phi::DenseTensor tensor_after_reshard;
  if (self->tensor.is_selected_rows()) {
    auto* selected_rows =
        static_cast<phi::SelectedRows*>(self->tensor.impl().get());
    ptr = static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
  } else if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
    auto* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
    PADDLE_ENFORCE(
        dist_tensor->initialized(),
        common::errors::Fatal(
            "The input dist tensor can't be uninitialized for we don't "
            "know the correct mesh to be reshard."));
    const auto& placements = dist_tensor->placements();
    bool need_reshard = false;
    for (const auto& placement : placements) {
      if (!placement->is_replicated()) {
        need_reshard = true;
        break;
      }
    }
    if (need_reshard) {
      tensor_after_reshard = ReshardXToReplicated(dist_tensor);
      ptr = &tensor_after_reshard;
    } else {
      ptr = dist_tensor->unsafe_mutable_value();
    }
#else
    PADDLE_THROW(common::errors::Unavailable(
        "The `_getitem_from_offset` method of (Dist)Tensor is not supported "
        "in the current PaddlePaddle, please recompile and install "
        "PaddlePaddle "
        "with the option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    ptr = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  }
  PADDLE_ENFORCE_NOT_NULL(ptr,
                          common::errors::InvalidArgument(
                              "%s is not a DenseTensor.", self->tensor.name()));
  const auto& tensor = *ptr;
  PADDLE_ENFORCE_EQ(
      tensor.IsInitialized(),
      true,
      common::errors::InvalidArgument(
          "Tensor of %s is Empty, please check if it has no data.",
          self->tensor.name()));

  const auto& tensor_dims = tensor.dims();

  std::vector<size_t> dims(tensor_dims.size());
  std::vector<size_t> stride = common::vectorize<size_t>(tensor.strides());

  size_t numel = 1;
  for (int i = tensor_dims.size() - 1; i >= 0; --i) {
    dims[i] = static_cast<size_t>(tensor_dims[i]);
    numel *= dims[i];
  }
  size_t offset = 0;
  if (PyTuple_Size(args) == 0) {
    PADDLE_ENFORCE_EQ(numel,
                      1,
                      common::errors::InvalidArgument(
                          "only one element tensors can be converted to Python "
                          "scalars when no input coordinates"));
  } else if (PyTuple_Size(args) == 1) {
    offset = CastPyArg2AttrLong(PyTuple_GET_ITEM(args, 0), 0);
    PADDLE_ENFORCE_LT(
        offset,
        numel,
        common::errors::InvalidArgument(
            "index %d is out of bounds for size %d", offset, numel));
  } else {
    PADDLE_ENFORCE_EQ(PyTuple_Size(args),
                      dims.size(),
                      common::errors::InvalidArgument(
                          "incorrect number of indices for Tensor"));

    for (Py_ssize_t i = 0; i < PyTuple_Size(args); ++i) {
      size_t index = CastPyArg2AttrLong(PyTuple_GET_ITEM(args, i), i);
      PADDLE_ENFORCE_LT(
          index,
          dims[i],
          common::errors::InvalidArgument(
              "index %d is out of bounds for axis %d with size %d",
              index,
              i,
              dims[i]));
      offset += index * stride[i];
    }
  }
#define PD_FOR_EACH_DENSE_TENSOR_DATA_TYPE(_) \
  _(bool, DataType::BOOL)                     \
  _(int8_t, DataType::INT8)                   \
  _(uint8_t, DataType::UINT8)                 \
  _(int16_t, DataType::INT16)                 \
  _(uint16_t, DataType::UINT16)               \
  _(int32_t, DataType::INT32)                 \
  _(uint32_t, DataType::UINT32)               \
  _(int64_t, DataType::INT64)                 \
  _(uint64_t, DataType::UINT64)               \
  _(bfloat16, DataType::BFLOAT16)             \
  _(float16, DataType::FLOAT16)               \
  _(float, DataType::FLOAT32)                 \
  _(double, DataType::FLOAT64)                \
  _(complex64, DataType::COMPLEX64)           \
  _(complex128, DataType::COMPLEX128)

#define TENSOR_TO_PY_SCALAR(T, proto_type)                                   \
  if (tensor.dtype() == proto_type) {                                        \
    auto numpy_dtype = TensorDtype2NumpyDtype(proto_type);                   \
    T b = paddle::pybind::TensorGetElement<T>(tensor, offset);               \
    Py_intptr_t py_dims[phi::DDim::kMaxRank];    /* NOLINT */                \
    Py_intptr_t py_strides[phi::DDim::kMaxRank]; /* NOLINT */                \
    auto& api = pybind11::detail::npy_api::get();                            \
    PyObject* array = api.PyArray_NewFromDescr_(                             \
        api.PyArray_Type_,                                                   \
        api.PyArray_DescrFromType_(numpy_dtype),                             \
        0,                                                                   \
        py_dims,                                                             \
        py_strides,                                                          \
        nullptr,                                                             \
        pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |                      \
            pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,                 \
        nullptr);                                                            \
    std::memcpy(                                                             \
        reinterpret_cast<void*>(pybind11::detail::array_proxy(array)->data), \
        static_cast<void*>(&b),                                              \
        sizeof(b));                                                          \
    return array;                                                            \
  }

  PD_FOR_EACH_DENSE_TENSOR_DATA_TYPE(TENSOR_TO_PY_SCALAR);
#undef TENSOR_TO_PY_SCALAR
  PADDLE_THROW(common::errors::Unimplemented("Unsupported tensor data type: %s",
                                             tensor.dtype()));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__setitem_dygraph(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  SetPythonStack();
  VLOG(4) << "Call new indexing strategy _setitem_dygraph";

  PyObject* _index = PyTuple_GET_ITEM(args, 0);
  PyObject* value_obj = PyTuple_GET_ITEM(args, 1);

  // NOTE(zhiqiu): PyTuple_Pack increases refcount while PyTuple_New
  // https://github.com/python/cpython/blob/24b63c695ae0a95b06379eaadace66735abac1e2/Objects/tupleobject.c#L251
  PyObject* index_ptr =
      !PyTuple_Check(_index) ? PyTuple_Pack(1, _index) : _index;
  DEFINE_PADDLE_SCOPE_GUARD([index_ptr, &_index]() {
    if (!PyTuple_Check(_index)) {
      Py_DECREF(index_ptr);
      VLOG(4) << "Call Py_DECREF";
    }
  });

  auto tensor = self->tensor;
  if (egr::Controller::Instance().HasGrad()) {
    PADDLE_ENFORCE_EQ(
        egr::EagerUtils::IsLeafTensor(tensor) &&
            !egr::EagerUtils::autograd_meta(&tensor)->StopGradient(),
        false,
        common::errors::InvalidArgument(
            "Leaf Tensor (%s) that doesn't stop gradient can't use "
            "inplace strategy.",
            tensor.name()));
  }
  const int rank = tensor.shape().size();
  std::vector<int> slice_starts, slice_ends, slice_strides;
  std::vector<int64_t> slice_axes, decrease_axis, infer_flags, none_axes;

  bool has_advanced_index = false;
  bool use_strided_slice = false;
  std::vector<int> advanced_index_dim(
      rank * 2,
      -1);  // content is dim, multiply 2 is to avoid all index are None
  std::vector<paddle::Tensor> advanced_index;  // content is index tensor

  // step1: parsing the index and recording them
  ParseIndex(tensor,
             _index,
             &slice_axes,
             &slice_starts,
             &slice_ends,
             &slice_strides,
             &decrease_axis,
             &none_axes,
             &infer_flags,
             &advanced_index_dim,
             &advanced_index,
             &has_advanced_index,
             &use_strided_slice);

  // step2: Parse values
  std::vector<phi::Scalar> values;
  paddle::Tensor value_tensor =
      dealWithValues(tensor, value_obj, &values, has_advanced_index);

  if (!has_advanced_index) {
    // use set_value OP if there is no advanced index

    // Release gil and do tracing
    py::gil_scoped_release release;
    // use inplace set_value_ operator
    if (value_tensor.initialized()) {
      if (self->tensor.dtype() != value_tensor.dtype()) {
        if (egr::Controller::Instance().GetAMPLevel() !=
            paddle::imperative::AmpLevel::O0) {
          paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
              tmps = {{self->tensor}, {value_tensor}};
          auto amp_dtype =
              paddle::imperative::GetAmpDestDtype("set_value", tmps);
          self->tensor = paddle::imperative::AmpAutoCast(
              self->tensor.name(), self->tensor, amp_dtype, "set_value");
          value_tensor = paddle::imperative::AmpAutoCast(
              value_tensor.name(), value_tensor, amp_dtype, "set_value");
        }
        if (self->tensor.dtype() != value_tensor.dtype()) {
          value_tensor = cast_ad_func(value_tensor, self->tensor.dtype());
        }
      }

      // step3.1: Only basic indexing, use OP set_value.
      const phi::distributed::ProcessMesh* mesh = nullptr;
      if (InputsContainDistTensor(&mesh, self->tensor, value_tensor)) {
        ConvertAllInputsToDistTensor(mesh, self->tensor, value_tensor);
      }
      self->tensor = set_value_with_tensor__ad_func(self->tensor,
                                                    value_tensor,
                                                    slice_starts,
                                                    slice_ends,
                                                    slice_strides,
                                                    slice_axes,
                                                    decrease_axis,
                                                    none_axes);
      if (PyCheckTensor(value_obj)) {
        // pass the stop_gradient from value to tensor.
        // pass stop gradient should be done after CheckInplace in
        // set_value__dygraph_function.
        if (!egr::EagerUtils::autograd_meta(&value_tensor)->StopGradient() &&
            egr::EagerUtils::autograd_meta(&self->tensor)->StopGradient()) {
          egr::EagerUtils::autograd_meta(&self->tensor)->SetStopGradient(false);
        }
      }
    } else {
      const phi::distributed::ProcessMesh* mesh = nullptr;
      if (InputsContainDistTensor(&mesh, self->tensor)) {
        ConvertAllInputsToDistTensor(mesh, self->tensor);
      }
      self->tensor = set_value__ad_func(self->tensor,
                                        slice_starts,
                                        slice_ends,
                                        slice_strides,
                                        slice_axes,
                                        decrease_axis,
                                        none_axes,
                                        {1},
                                        values);
    }
  } else {
    // step3.2: Case for there are advanced indexing.
    //   1. get __getitem__ result of basic indexing;
    //   2. transpose original tensor so that the axis with advanced indexing
    //   will come to the first;
    //   3. assign values to the sliced result by index_put OP;
    //   4. transpose back and assign the result to original tensor by set_value
    //   OP.
    bool out_is_view = false;
    paddle::Tensor sub_tensor = getTensorWithBasicIndexing(tensor,
                                                           &slice_axes,
                                                           &slice_starts,
                                                           &slice_ends,
                                                           &slice_strides,
                                                           &decrease_axis,
                                                           &none_axes,
                                                           &infer_flags,
                                                           &use_strided_slice,
                                                           &out_is_view);

    std::vector<paddle::Tensor> transed_index;
    std::vector<int> trans_back_dim, trans_dim;

    int pos_of_new_dim = INT_MAX, rank_of_new_dim = 1;

    paddle::Tensor transed_sub_tensor =
        dealWithAdvancedIndex(sub_tensor,
                              &advanced_index_dim,
                              &advanced_index,
                              true,
                              &transed_index,
                              &trans_back_dim,
                              &pos_of_new_dim,
                              &rank_of_new_dim,
                              &trans_dim,
                              &out_is_view);

    // Release gil and do tracing
    py::gil_scoped_release release;
    if (value_tensor.initialized()) {
      if (self->tensor.dtype() != value_tensor.dtype()) {
        if (egr::Controller::Instance().GetAMPLevel() !=
            paddle::imperative::AmpLevel::O0) {
          paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
              tmps = {{self->tensor}, {value_tensor}};
          auto amp_dtype =
              paddle::imperative::GetAmpDestDtype("index_put", tmps);
          self->tensor = paddle::imperative::AmpAutoCast(
              self->tensor.name(), self->tensor, amp_dtype, "index_put");
          value_tensor = paddle::imperative::AmpAutoCast(
              value_tensor.name(), value_tensor, amp_dtype, "index_put");
        }
        if (self->tensor.dtype() != value_tensor.dtype()) {
          value_tensor = cast_ad_func(value_tensor, self->tensor.dtype());
        }
      }

      if (value_tensor.dims().size() > 1 && pos_of_new_dim != 0) {
        value_tensor = transpose_ad_func(value_tensor, trans_dim);
      }

      const phi::distributed::ProcessMesh* mesh = nullptr;
      if (InputsContainDistTensor(
              &mesh, self->tensor, transed_sub_tensor, value_tensor)) {
        ConvertAllInputsToDistTensor(
            mesh, self->tensor, transed_sub_tensor, value_tensor);
      }
      if (transed_index.size() == 1 &&
          transed_index[0].dtype() == phi::DataType::BOOL &&
          transed_index[0].shape().size() == self->tensor.shape().size() &&
          value_tensor.numel() == 1) {
        value_tensor = expand_ad_func(value_tensor, self->tensor.shape());
        transed_sub_tensor =
            where__ad_func(logical_not_ad_func(transed_index[0]),
                           transed_sub_tensor,
                           value_tensor);
      } else {
        transed_sub_tensor =
            index_put__ad_func(transed_sub_tensor, transed_index, value_tensor);
      }
      if (out_is_view) {
        // NOTE(zoooo0820): if out_is_view is true, it is a case of
        // combined-indexing setitem, i.e. firstly we get a view of
        // self->tensor, then modified it with inplace api index_put_ For now,
        // in design of Paddle, the forward result is right. But the backward
        // edge can not be established because the Base Tensor cannot sense
        // whether it has been modified by other operations. Following codes are
        // to add a new node (set_value_with_tensor_grad) to record the backward
        // edge, with out ad_function which needs to do the forward calculation.

        egr::AutogradMeta* x_autograd_meta =
            egr::EagerUtils::nullable_autograd_meta(self->tensor);
        egr::AutogradMeta* values_autograd_meta =
            egr::EagerUtils::nullable_autograd_meta(transed_sub_tensor);
        bool trace_backward = egr::Controller::Instance().HasGrad();
        bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(
            trace_backward, x_autograd_meta, values_autograd_meta);
        // Node Declaration
        std::shared_ptr<SetValueWithTensorGradNode> grad_node;
        // Set grad_node before API Call
        if (require_any_grad) {
          paddle::Tensor transback_sub_tensor =
              transpose_ad_func(transed_sub_tensor, trans_back_dim);

          const auto& values_tmp =
              (require_any_grad && transback_sub_tensor.is_dense_tensor() &&
               !std::dynamic_pointer_cast<phi::DenseTensor>(
                    transback_sub_tensor.impl())
                    ->meta()
                    .is_contiguous())
                  ? paddle::Tensor(
                        std::make_shared<phi::DenseTensor>(
                            paddle::experimental::Trans2Contiguous(
                                *(std::dynamic_pointer_cast<phi::DenseTensor>(
                                    transback_sub_tensor.impl())))),
                        transback_sub_tensor.mutable_autograd_meta(),
                        transback_sub_tensor.name())
                  : transback_sub_tensor;
          if (!x_autograd_meta) {
            VLOG(3) << "x_autograd_meta is null and requires_any_grad is true";
            x_autograd_meta = egr::EagerUtils::autograd_meta(&self->tensor);
          }
          grad_node = std::shared_ptr<SetValueWithTensorGradNode>(
              new SetValueWithTensorGradNode(1, 2));  // NOLINT
          grad_node->SetAttribute_starts(slice_starts);
          grad_node->SetAttribute_ends(slice_ends);
          grad_node->SetAttribute_steps(slice_strides);
          grad_node->SetAttribute_axes(slice_axes);
          grad_node->SetAttribute_decrease_axes(decrease_axis);
          grad_node->SetAttribute_none_axes(none_axes);
          grad_node->SetTensorWrapper_values(values_tmp);

          paddle::memory::LogDeviceMemoryStats(
              egr::Controller::Instance().GetExpectedPlace(),
              "set_value_with_tensor");
          egr::EagerUtils::CheckInplace(
              self->tensor, x_autograd_meta, require_any_grad);
          egr::EagerUtils::PassStopGradient(false, x_autograd_meta);
          // SetGradOutMeta & SetEdges
          grad_node->SetGradOutMeta(self->tensor, 0);
          grad_node->SetGradOutMeta(transback_sub_tensor, 1);
          grad_node->SetGradInMeta(self->tensor, 0);

          egr::EagerUtils::SetOutRankWithSlot(x_autograd_meta, 0);
          egr::EagerUtils::SetHistory(x_autograd_meta, grad_node);
        }
      } else {
        self->tensor.set_autograd_meta(
            transed_sub_tensor.mutable_autograd_meta());
      }
      if (PyCheckTensor(value_obj)) {
        // pass the stop_gradient from value to tensor.
        // pass stop gradient should be done after CheckInplace in
        // set_value__dygraph_function.
        if (!egr::EagerUtils::autograd_meta(&value_tensor)->StopGradient() &&
            egr::EagerUtils::autograd_meta(&self->tensor)->StopGradient()) {
          egr::EagerUtils::autograd_meta(&self->tensor)->SetStopGradient(false);
        }
      }
    }
  }

  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_apply(TensorObject* self,
                              PyObject* args,
                              PyObject* kwargs) {
  EAGER_TRY
  PyObject* apply_func = PyTuple_GET_ITEM(args, 0);
  PyTensorHook func = PyTensorHook(apply_func);
  paddle::Tensor out = func(self->tensor);
  return ToPyObject(out);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_apply_(TensorObject* self,
                               PyObject* args,
                               PyObject* kwargs) {
  EAGER_TRY
  PyObject* apply_func = PyTuple_GET_ITEM(args, 0);
  PyTensorHook func = PyTensorHook(apply_func);
  paddle::Tensor out = func(self->tensor);
  self->tensor.set_impl(out.impl());
  Py_INCREF(self);
  return reinterpret_cast<PyObject*>(self);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_register_grad_hook(TensorObject* self,
                                           PyObject* args,
                                           PyObject* kwargs) {
  EAGER_TRY
  SetPythonStack();
  int64_t hook_id = 0;
  if (egr::EagerUtils::IsLeafTensor(self->tensor)) {
    VLOG(6) << "Register hook for leaf tensor: " << self->tensor.name();

    auto autograd_meta = egr::EagerUtils::unsafe_autograd_meta(self->tensor);

    if (autograd_meta && !autograd_meta->StopGradient()) {
      if (!autograd_meta->GetMutableGradNode()) {
        VLOG(6) << "Detected nullptr grad_node, Leaf tensor should have had "
                   "grad_node with type: GradNodeAccumulation.";
        autograd_meta->SetGradNode(
            std::make_shared<egr::GradNodeAccumulation>(autograd_meta));
      }
    }

    std::shared_ptr<egr::GradNodeBase> grad_node =
        egr::EagerUtils::grad_node(self->tensor);
    auto rank_info =
        egr::EagerUtils::unsafe_autograd_meta(self->tensor)->OutRankInfo();
    PyObject* hook_func = PyTuple_GET_ITEM(args, 0);

    auto accumulation_grad_node =
        std::dynamic_pointer_cast<egr::GradNodeAccumulation>(grad_node);
    hook_id = accumulation_grad_node->RegisterGradientHook(
        rank_info.first,
        rank_info.second,
        std::make_shared<PyTensorHook>(hook_func));

  } else {
    VLOG(6) << "Register hook for non leaf tensor: " << self->tensor.name();
    std::shared_ptr<egr::GradNodeBase> grad_node =
        egr::EagerUtils::grad_node(self->tensor);
    auto rank_info =
        egr::EagerUtils::unsafe_autograd_meta(self->tensor)->OutRankInfo();

    PyObject* hook_func = PyTuple_GET_ITEM(args, 0);

    hook_id = grad_node->RegisterGradientHook(
        rank_info.first,
        rank_info.second,
        std::make_shared<PyTensorHook>(hook_func));
  }
  return ToPyObject(hook_id);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_remove_grad_hook(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  VLOG(6) << "Remove the registered hook for tensor: " << self->tensor.name();
  std::shared_ptr<egr::GradNodeBase> grad_node =
      egr::EagerUtils::grad_node(self->tensor);

  int64_t hook_id = pybind::CastPyArg2AttrLong(PyTuple_GET_ITEM(args, 0), 0);

  return ToPyObject(grad_node->RemoveGradientHook(hook_id));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_inplace_assign(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  VLOG(6) << "inplace assign for tensor:" << self->tensor.name();
  PyObject* other = PyTuple_GET_ITEM(args, 0);
  PyObject* self_obj = reinterpret_cast<PyObject*>(self);
  ShareTensor(self_obj, other);
  RETURN_PY_NONE;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method__register_reduce_hook__doc__,  // NOLINT
             R"DOC(_register_backward_hook($self, hook, /)
--

Registers a backward hook for current Tensor.

This hook will be called every time the gradient of current Tensor has been fully calculated.

There are two differences with `_register_grad_hook`:
1. This backward hook will be executed after the gradient accumulation completed across batches,
  but the hook registered by `_register_grad_hook` will be executed before the gradient accumulation
  completed in current batch.
2. This backward hook function should have the following signature:

    hook() -> None

  It requires no input and no return value.

Args:
    hook(function): A backward hook to be registered for Tensor.gradient

Returns:
    None
)DOC");
static PyObject* tensor_register_reduce_hook(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Register reduce hook for tensor: " << self->tensor.name();

  std::shared_ptr<egr::GradNodeBase> grad_node =
      egr::EagerUtils::grad_node(self->tensor);
  PADDLE_ENFORCE_EQ(egr::EagerUtils::IsLeafTensor(self->tensor),
                    true,
                    common::errors::InvalidArgument(
                        "Only can register backward hook for leaf Tensor."));
  PADDLE_ENFORCE_EQ(
      !egr::EagerUtils::unsafe_autograd_meta(self->tensor)->StopGradient(),
      true,
      common::errors::InvalidArgument(
          "Cannot register backward hook on a Tensor that stop "
          "gradient."));
  PADDLE_ENFORCE(grad_node.get() != nullptr,
                 common::errors::Fatal("Detected nullptr grad_node,"
                                       "Leaf tensor should have had grad_node "
                                       "with type: GradNodeAccumulation."));
  PyObject* hook_func = PyTuple_GET_ITEM(args, 0);

  auto accumulation_grad_node =
      std::dynamic_pointer_cast<egr::GradNodeAccumulation>(grad_node);
  accumulation_grad_node->RegisterReduceHook(
      std::make_shared<PyVoidHook>(hook_func));

  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__set_grad_type(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  auto var_type = pybind::CastPyArg2ProtoType(PyTuple_GET_ITEM(args, 0), 0);
  auto grad_tensor =
      egr::EagerUtils::autograd_meta(&self->tensor)->MutableGrad();
  if (var_type == framework::proto::VarType::DENSE_TENSOR) {
    grad_tensor->set_impl(std::make_shared<phi::DenseTensor>());
  } else if (var_type == framework::proto::VarType::SELECTED_ROWS) {
    grad_tensor->set_impl(std::make_shared<phi::SelectedRows>());
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__clear(TensorObject* self,
                               PyObject* args,
                               PyObject* kwargs) {
  EAGER_TRY
  self->tensor.reset();
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__clear_dataptr(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  self->tensor.set_impl(nullptr);
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__copy_gradient_from(TensorObject* self,
                                            PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  auto src = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  if (self->tensor.initialized()) {
    PADDLE_ENFORCE_EQ(self->tensor.dtype(),
                      src.dtype(),
                      common::errors::PreconditionNotMet(
                          "Tensor %s has different data type with Tensor %s",
                          self->tensor.name(),
                          src.name()));
    PADDLE_ENFORCE_EQ(self->tensor.impl()->type_info().id(),
                      src.impl()->type_info().id(),
                      common::errors::PreconditionNotMet(
                          "Tensor %s has different type with Tensor %s, Tensor "
                          "ShareGradientDataWith cannot be performed!",
                          self->tensor.name(),
                          src.name()));
  }
  VLOG(6) << "Tensor copy gradient from: " << src.name();
  auto* p_grad = egr::EagerUtils::mutable_grad(self->tensor);
  if (p_grad) {
    PADDLE_ENFORCE_EQ(src.initialized(),
                      true,
                      common::errors::InvalidArgument(
                          "Tensor %s has not been initialized", src.name()));
    p_grad->set_impl(src.impl());
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__use_gpudnn(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.defined() && (self->tensor.is_dense_tensor() ||
                                            self->tensor.is_dist_tensor()),
                 common::errors::Fatal("Function _use_gpudnn is only effective "
                                       "for DenseTensor and DistTensor."));

  bool use_gpudnn = pybind::CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 0), 0);

  // Set the same use_gpudnn attribute, return directly
  phi::DenseTensor* dense_tensor = nullptr;
  if (self->tensor.is_dist_tensor()) {
    dense_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get())
            ->unsafe_mutable_value();
  } else {
    dense_tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  }

  phi::DenseTensorMeta* dense_tensor_meta =
      phi::DenseTensorUtils::GetMutableMeta(dense_tensor);
  if (use_gpudnn == dense_tensor_meta->use_gpudnn) {
    return ToPyObject(self->tensor);
  }

  // Share all other members of Tensor except use_gpudnn
  phi::DenseTensorMeta target_dense_meta = *dense_tensor_meta;
  target_dense_meta.use_gpudnn = use_gpudnn;
  phi::DenseTensor target_dense_tensor;
  target_dense_tensor.ShareDataWith(*dense_tensor);
  target_dense_tensor.set_meta(target_dense_meta);
  // Construct returned tensor
  paddle::Tensor target_tensor(self->tensor.name());
  target_tensor.set_autograd_meta(self->tensor.mutable_autograd_meta());
  if (self->tensor.is_dist_tensor()) {
    auto dist_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
    auto target_dist_tensor = std::make_shared<phi::distributed::DistTensor>(
        dist_tensor->dims(), dist_tensor->dist_attr());
    *(target_dist_tensor->unsafe_mutable_value()) = target_dense_tensor;
    target_tensor.set_impl(target_dist_tensor);
  } else {
    target_tensor.set_impl(
        std::make_shared<phi::DenseTensor>(target_dense_tensor));
  }

  VLOG(4) << "Tensor: " << target_tensor.name()
          << " set use_gpudnn = " << use_gpudnn;

  return ToPyObject(target_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_set_vocab(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  using Vocab = phi::Vocab;
  auto vocab = CastPyArg2Vocab(PyTuple_GET_ITEM(args, 0), 0);
  auto var_tensor = std::make_shared<egr::VariableCompatTensor>();
  *var_tensor->GetMutable<Vocab>() = vocab;
  self->tensor.set_impl(var_tensor);
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_set_string_list(TensorObject* self,
                                               PyObject* args,
                                               PyObject* kwargs) {
  EAGER_TRY
  using Strings = phi::Strings;
  auto strings = CastPyArg2VectorOfString(PyTuple_GET_ITEM(args, 0), 0);
  auto var_tensor = std::make_shared<egr::VariableCompatTensor>();
  *var_tensor->GetMutable<Strings>() = strings;
  self->tensor.set_impl(var_tensor);
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_get_map_tensor(TensorObject* self,
                                              PyObject* args,
                                              PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE_EQ(
      egr::IsVariableCompatTensor(self->tensor),
      true,
      common::errors::Fatal(
          "this method is only effective for VariableCompatTensor"));
  using Vocab = phi::Vocab;
  auto* var_tensor =
      static_cast<const egr::VariableCompatTensor*>(self->tensor.impl().get());
  return ToPyObject(var_tensor->Get<Vocab>());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_nnz__doc__,
             R"DOC(nnz($self, /)
--

Note:
    **This API is only available for SparseCooTensor or SparseCsrTensor.**

Returns the total number of non zero elements in input SparseCooTensor/SparseCsrTensor.

Returns:
    int

Examples:

    .. code-block:: python

        >>> import paddle

        >>> indices = [[0, 1, 2], [1, 2, 0]]
        >>> values = [1.0, 2.0, 3.0]
        >>> dense_shape = [3, 3]
        >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        >>> coo.nnz()
        3

)DOC");  // NOLINT

static PyObject* tensor_method_get_non_zero_nums(TensorObject* self,
                                                 PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_coo_tensor() ||
                     self->tensor.is_sparse_csr_tensor(),
                 common::errors::Fatal("this method is only effective for "
                                       "SparseCooTensor or SparseCsrTensor"));
  if (self->tensor.is_sparse_coo_tensor()) {
    auto sparse_coo_tensor =
        std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
    return ToPyObject(sparse_coo_tensor->nnz());
  } else {
    auto sparse_csr_tensor =
        std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
    return ToPyObject(sparse_csr_tensor->nnz());
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_indices__doc__,
             R"DOC(indices($self, /)
--

Note:
    **This API is only available for SparseCooTensor.**

Returns the indices of non zero elements in input SparseCooTensor.

Returns:
    DenseTensor

Examples:

    .. code-block:: python

        >>> import paddle

        >>> indices = [[0, 1, 2], [1, 2, 0]]
        >>> values = [1.0, 2.0, 3.0]
        >>> dense_shape = [3, 3]
        >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        >>> coo.indices()
        Tensor(shape=[2, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
        [[0, 1, 2],
         [1, 2, 0]])

)DOC");  // NOLINT

static PyObject* tensor_method_get_non_zero_indices(TensorObject* self,
                                                    PyObject* args,
                                                    PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_coo_tensor(),
                 common::errors::Fatal(
                     "this method is only effective for SparseCooTensor"));
  auto sparse_coo_tensor =
      std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
  paddle::Tensor tensor(std::make_shared<phi::DenseTensor>(
      sparse_coo_tensor->non_zero_indices()));
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_values__doc__,
             R"DOC(values($self, /)
--

Note:
    **This API is only available for SparseCooTensor or SparseCsrTensor.**

Returns the values of non zero elements in input SparseCooTensor.

Returns:
    DenseTensor

Examples:

    .. code-block:: python

        >>> import paddle

        >>> indices = [[0, 1, 2], [1, 2, 0]]
        >>> values = [1.0, 2.0, 3.0]
        >>> dense_shape = [3, 3]
        >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        >>> coo.values()
        Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
        [1., 2., 3.])

)DOC");  // NOLINT

static PyObject* tensor_method_get_non_zero_elements(TensorObject* self,
                                                     PyObject* args,
                                                     PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_coo_tensor() ||
                     self->tensor.is_sparse_csr_tensor(),
                 common::errors::Fatal("this method is only effective for "
                                       "SparseCooTensor or SparseCsrTensor"));
  if (self->tensor.is_sparse_coo_tensor()) {
    auto sparse_coo_tensor =
        std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
    paddle::Tensor tensor(std::make_shared<phi::DenseTensor>(
        sparse_coo_tensor->non_zero_elements()));
    return ToPyObject(tensor);
  } else {
    auto sparse_csr_tensor =
        std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
    paddle::Tensor tensor(std::make_shared<phi::DenseTensor>(
        sparse_csr_tensor->non_zero_elements()));
    return ToPyObject(tensor);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_crows__doc__,
             R"DOC(crows($self, /)
--

Note:
    **This API is only available for SparseCsrTensor.**

Returns the compressed row index of non zero elements in input SparseCsrTensor.

Returns:
    DenseTensor

Examples:

    .. code-block:: python

        >>> import paddle

        >>> crows = [0, 2, 3, 5]
        >>> cols = [1, 3, 2, 0, 1]
        >>> values = [1, 2, 3, 4, 5]
        >>> dense_shape = [3, 4]
        >>> csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
        >>> csr.crows()
        Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
        [0, 2, 3, 5])

)DOC");  // NOLINT

static PyObject* tensor_method_get_non_zero_crows(TensorObject* self,
                                                  PyObject* args,
                                                  PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_csr_tensor(),
                 common::errors::Fatal(
                     "this method is only effective for SparseCsrTensor"));
  auto sparse_csr_tensor =
      std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
  paddle::Tensor tensor(
      std::make_shared<phi::DenseTensor>(sparse_csr_tensor->non_zero_crows()));
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_cols__doc__,
             R"DOC(cols($self, /)
--

Note:
    **This API is only available for SparseCsrTensor.**

Returns the column index of non zero elements in input SparseCsrTensor.

Returns:
    DenseTensor

Examples:

    .. code-block:: python

        >>> import paddle

        >>> crows = [0, 2, 3, 5]
        >>> cols = [1, 3, 2, 0, 1]
        >>> values = [1, 2, 3, 4, 5]
        >>> dense_shape = [3, 4]
        >>> csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
        >>> csr.cols()
        Tensor(shape=[5], dtype=int64, place=Place(gpu:0), stop_gradient=True,
        [1, 3, 2, 0, 1])

)DOC");  // NOLINT

static PyObject* tensor_method_get_non_zero_cols(TensorObject* self,
                                                 PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_csr_tensor(),
                 common::errors::Fatal(
                     "this method is only effective for SparseCsrTensor"));
  auto sparse_csr_tensor =
      std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
  paddle::Tensor tensor(
      std::make_shared<phi::DenseTensor>(sparse_csr_tensor->non_zero_cols()));
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_is_dense__doc__, R"DOC(is_dense($self, /)
--

Whether the Tensor is a Dense Tensor.

Returns:
    Whether the Tensor is a Dense Tensor.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1.0], stop_gradient=False)
        >>> print(x.is_dense())
        True
)DOC");  // NOLINT

static PyObject* tensor_method_is_dense(TensorObject* self,
                                        PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_dense_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_is_dist__doc__, R"DOC(is_dist($self, /)
--

Whether the Tensor is a Distributed Tensor.

Returns:
    Whether the Tensor is a Distributed Tensor.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1.0], stop_gradient=False)
        >>> print(x.is_dist())
        False
)DOC");  // NOLINT

static PyObject* tensor_method_is_dist(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_dist_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_sparse__doc__,  // NOLINT
             R"DOC(is_sparse($self, /)
--

Returns whether the input Tensor is SparseCooTensor or SparseCsrTensor.

When input is SparseCooTensor/SparseCsrTensor, will return True. When input is DenseTensor, will return False.

Returns:
    bool

Examples:

    .. code-block:: python

        >>> import paddle

        >>> indices = [[0, 1, 2], [1, 2, 0]]
        >>> values = [1.0, 2.0, 3.0]
        >>> dense_shape = [3, 3]
        >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        >>> coo.is_sparse()
        True

)DOC");                                // NOLINT

static PyObject* tensor_method_is_sparse(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_sparse_coo_tensor() ||
                    self->tensor.is_sparse_csr_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_sparse_coo__doc__,  // NOLINT
             R"DOC(is_sparse_coo($self, /)
--

Returns whether the input Tensor is SparseCooTensor.

When input is SparseCooTensor, will return True. When input is DenseTensor/SparseCsrTensor, will return False.

Returns:
    bool

Examples:

    .. code-block:: python

        >>> import paddle

        >>> indices = [[0, 1, 2], [1, 2, 0]]
        >>> values = [1.0, 2.0, 3.0]
        >>> dense_shape = [3, 3]
        >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        >>> coo.is_sparse_coo()
        True

)DOC");                                    // NOLINT

static PyObject* tensor_method_is_sparse_coo(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_sparse_coo_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_sparse_csr__doc__,  // NOLINT
             R"DOC(is_sparse_csr($self, /)
--

Returns whether the input Tensor is SparseCsrTensor.

When input is SparseCsrTensor, will return True. When input is DenseTensor/SparseCooTensor, will return False.

Returns:
    bool

Examples:

    .. code-block:: python

        >>> import paddle

        >>> crows = [0, 2, 3, 5]
        >>> cols = [1, 3, 2, 0, 1]
        >>> values = [1, 2, 3, 4, 5]
        >>> dense_shape = [3, 4]
        >>> csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
        >>> csr.is_sparse_csr()
        True

)DOC");                                    // NOLINT

static PyObject* tensor_method_is_sparse_csr(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_sparse_csr_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_to_sparse_csr__doc__,  // NOLINT
             R"DOC(to_sparse_csr($self, /)
--

Note:
    **This API is only available for DenseTensor or SparseCooTensor.**

Convert input Tensor to SparseCsrTensor.

When input is SparseCooTensor, will convert `COO` to `CSR` . When input is DenseTensor, will convert `Dense` to `CSR` .

Returns:
    SparseCsrTensor

Examples:

    .. code-block:: python

        >>> import paddle

        >>> indices = [[0, 1, 2], [1, 2, 0]]
        >>> values = [1.0, 2.0, 3.0]
        >>> dense_shape = [3, 3]
        >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        >>> coo.to_sparse_csr()
        Tensor(shape=[3, 3], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
        crows=[0, 1, 2, 3],
        cols=[1, 2, 0],
        values=[1., 2., 3.])

)DOC");                                    // NOLINT

static PyObject* tensor_method_to_sparse_csr(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  auto csr_tensor = self->tensor.to_sparse_csr();
  egr::EagerUtils::autograd_meta(&csr_tensor)
      ->SetStopGradient(
          egr::EagerUtils::autograd_meta(&self->tensor)->StopGradient());
  egr::EagerUtils::autograd_meta(&csr_tensor)
      ->SetPersistable(
          egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  return ToPyObject(csr_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_coalesced__doc__,  // NOLINT
             R"DOC(is_coalesced($self, /)
--

Check whether the Tensor is a coalesced SparseCooTensor. If not it will return False.
Tensor types other than SparseCooTensor are not supported.

Notes:
    It will return always False for a newly created SparseCooTensor.

Args:
    x (Tensor): The input tensor. It can only be SparseCooTensor.

Returns:
    bool: True if the Tensor is a coalesced SparseCooTensor, and False otherwise.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> indices = [[0, 0, 1], [1, 1, 2]]
        >>> values = [1.0, 2.0, 3.0]
        >>> x = paddle.sparse.sparse_coo_tensor(indices, values)

        >>> x.is_coalesced()
        False
        >>> x = x.coalesce()
        >>> x.is_coalesced()
        True

        >>> indices = [[0, 1, 1], [1, 0, 2]]
        >>> values = [1.0, 2.0, 3.0]
        >>> x = paddle.sparse.sparse_coo_tensor(indices, values)
        >>> x.is_coalesced()
        False

)DOC");                                   // NOLINT

static PyObject* tensor_method_is_coalesced(TensorObject* self,
                                            PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.is_sparse_coo_tensor()) {
    auto sparse_coo_tensor =
        std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
    return ToPyObject(sparse_coo_tensor->coalesced());
  } else {
    PADDLE_THROW(common::errors::InvalidType(
        "Method is_coalesced only support sparse coo tensor."));
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_same_shape__doc__,  // NOLINT
             R"DOC(is_same_shape($self, y, /)
--

Return the results of shape comparison between two Tensors, check whether x.shape equal to y.shape.
Any two type Tensor among DenseTensor/SparseCooTensor/SparseCsrTensor are supported.

Args:
    x (Tensor): The input tensor. It can be DenseTensor/SparseCooTensor/SparseCsrTensor.
    y (Tensor): The input tensor. It can be DenseTensor/SparseCooTensor/SparseCsrTensor.

Returns:
    bool: True for same shape and False for different shape.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.rand([2, 3, 8])
        >>> y = paddle.rand([2, 3, 8])
        >>> y = y.to_sparse_csr()
        >>> z = paddle.rand([2, 5])

        >>> x.is_same_shape(y)
        True
        >>> x.is_same_shape(z)
        False

)DOC");                                    // NOLINT

static PyObject* tensor_method_is_same_shape(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  auto other = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  return ToPyObject(self->tensor.shape() == other.shape());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__inplace_version(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  uint32_t inplace_version = self->tensor.current_inplace_version();

  return ToPyObject(inplace_version);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_element_size__doc__,  // NOLINT
             R"DOC(element_size($self, /)
--

Returns the size in bytes of an element in the Tensor.

Returns:
    int, The size in bytes of an element in the Tensor.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor(1, dtype='bool')
        >>> x.element_size()
        1

        >>> x = paddle.to_tensor(1, dtype='float16')
        >>> x.element_size()
        2

        >>> x = paddle.to_tensor(1, dtype='float32')
        >>> x.element_size()
        4

        >>> x = paddle.to_tensor(1, dtype='float64')
        >>> x.element_size()
        8

        >>> x = paddle.to_tensor(1, dtype='complex128')
        >>> x.element_size()
        16
)DOC");

static PyObject* tensor_method_element_size(TensorObject* self,
                                            PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  uint32_t element_size = phi::SizeOf(self->tensor.dtype());

  return ToPyObject(element_size);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method__bump_inplace_version__doc__,  // NOLINT
             R"DOC(_bump_inplace_version($self, /)
--

Note:
    **This API is ONLY available in Dygraph mode.**
    **This is a very low level API. Users should not use it directly. **

  Bump the version whenever the Tensor is modified through an inplace operation.
)DOC");
static PyObject* tensor__bump_inplace_version(TensorObject* self,
                                              PyObject* args,
                                              PyObject* kwargs) {
  EAGER_TRY
  self->tensor.bump_inplace_version();
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_is_selected_rows(TensorObject* self,
                                                PyObject* args,
                                                PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_selected_rows());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_get_rows(TensorObject* self,
                                        PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(
      self->tensor.is_selected_rows(),
      common::errors::Fatal("this method is only effective for SelectedRows"));
  auto selected_rows =
      std::dynamic_pointer_cast<phi::SelectedRows>(self->tensor.impl());
  return ToPyObject(selected_rows->rows());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__reset_grad_inplace_version(TensorObject* self,
                                                    PyObject* args,
                                                    PyObject* kwargs) {
  EAGER_TRY
  Py_ssize_t args_num = PyTuple_Size(args);
  bool set_to_zero = true;
  if (args_num == (Py_ssize_t)1) {
    set_to_zero = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 0), 0);
  }

  paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
  if (grad && grad->defined() && grad->initialized() &&
      (grad->is_dense_tensor() || grad->is_dist_tensor())) {
    grad->reset_inplace_version(set_to_zero);
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__share_memory(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
#ifndef _WIN32
  PADDLE_ENFORCE_EQ(phi::is_cpu_place(self->tensor.place()),
                    true,
                    common::errors::InvalidArgument(
                        "Sharing memory only support CPU Tensor currently"));
  // 1. get DenseTensor
  auto* t =
      std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl()).get();
  // 2. allocate shared memory
  void* data_ptr = t->data();
  size_t data_size =
      t->numel() *
      framework::SizeOfType(framework::TransToProtoVarType(t->dtype()));
  auto shared_writer_holder =
      memory::allocation::AllocateMemoryMapWriterAllocation(data_size);
  // 3. maintain mmap fd set & backup ipc_name
  const std::string& ipc_name = shared_writer_holder->ipc_name();
  memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);
  // 4. copy data & reset holder
  memory::Copy(phi::CPUPlace(),
               shared_writer_holder->ptr(),
               phi::CPUPlace(),
               data_ptr,
               data_size);
  t->ResetHolder(shared_writer_holder);
  return ToPyObject(t);
#else
  PADDLE_THROW(common::errors::PermissionDenied(
      "Sharing memory in Windows OS is not supported currently"));
  RETURN_PY_NONE

#endif
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__offset(TensorObject* self,
                                PyObject* args,
                                PyObject* kwargs) {
  EAGER_TRY
  phi::DenseTensor* dense_tensor = nullptr;
  if (self->tensor.is_dist_tensor()) {
    dense_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get())
            ->unsafe_mutable_value();
  } else {
    dense_tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  }
  PADDLE_ENFORCE_EQ(
      dense_tensor->IsInitialized(),
      true,
      common::errors::InvalidArgument("Tensor %s has not been initialized!",
                                      self->tensor.name()));

  return ToPyObject(dense_tensor->offset());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__grad_name(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE_EQ(
      grad != nullptr,
      true,
      common::errors::InvalidArgument(
          "Detected nullptr grad. Please check if you have manually "
          "cleared the grad inside autograd_meta"));
  return ToPyObject(grad->name());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__grad_value(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE_EQ(
      grad != nullptr,
      true,
      common::errors::InvalidArgument(
          "Detected nullptr grad. Please check if you have manually "
          "cleared the grad inside autograd_meta"));

  if (!grad->defined()) {
    RETURN_PY_NONE
  }
  if (grad->is_dense_tensor()) {
    auto* grad_tensor = static_cast<phi::DenseTensor*>(grad->impl().get());
    return ToPyObject(grad_tensor);
  } else if (grad->is_dist_tensor()) {
    auto* grad_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get())
            ->unsafe_mutable_value();
    return ToPyObject(grad_tensor);
  } else {
    PADDLE_THROW(common::errors::Fatal(
        "This method is only supported for DenseTensor and DistTensor."));
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__local_value(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
    paddle::Tensor result(
        std::make_shared<phi::DenseTensor>(dist_tensor->value()));
    return ToPyObject(result);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "The `_local_value` method of (Dist)Tensor is not supported "
        "in the current PaddlePaddle, please recompile and install "
        "PaddlePaddle "
        "with the option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__unset_fake_empty(TensorObject* self,
                                          PyObject* args,
                                          PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE_EQ(
      grad != nullptr,
      true,
      common::errors::InvalidArgument(
          "Detected nullptr grad. Please check if you have manually "
          "cleared the grad inside autograd_meta"));

  bool is_leaf = egr::EagerUtils::IsLeafTensor(self->tensor);
  if (is_leaf) {
    std::static_pointer_cast<egr::GradNodeAccumulation>(
        egr::EagerUtils::grad_node(self->tensor))
        ->SetFakeEmpty(false);
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_data_ptr__doc__,  // NOLINT
             R"DOC(data_ptr($self, /)
--

Returns the address of the first element of current Tensor.

Returns:
    int, The address of the first element of current Tensor.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1, 2, 3])
        >>> print(x.data_ptr())
        >>> # doctest: +SKIP('return the address')
        93220864
        >>> # doctest: -SKIP
)DOC");                               // NOLINT

static PyObject* tensor_data_ptr(TensorObject* self,
                                 PyObject* args,
                                 PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.defined() && self->tensor.has_allocation() &&
      self->tensor.is_dense_tensor()) {
    return ToPyObject(
        (int64_t)std::dynamic_pointer_cast<phi::DenseTensor>(  // NOLINT
            self->tensor.impl())
            ->data());
  } else if (self->tensor.defined() && self->tensor.has_allocation() &&
             self->tensor.is_dist_tensor()) {
    return ToPyObject(
        (int64_t)
            std::dynamic_pointer_cast<phi::distributed::DistTensor>(  // NOLINT
                self->tensor.impl())
                ->unsafe_mutable_value()
                ->data());
  }

  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__grad_ivar(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  EAGER_TRY
  VLOG(6) << "Get grad for tensor: " << self->tensor.name();
  auto meta = egr::EagerUtils::nullable_autograd_meta(self->tensor);
  VLOG(6) << meta << " initialized: " << meta->Grad().initialized();
  if (meta && meta->Grad().initialized()) {
    return ToPyObject(meta->Grad());
  } else {
    if (meta && !meta->Grad().initialized() && meta->Grad().impl() &&
        meta->Grad().is_dist_tensor()) {
      return ToPyObject(meta->Grad(), false);
    }
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_get_strides__doc__,  // NOLINT
             R"DOC(get_strides($self, /)
--

Returns the strides of current Tensor.

Returns:
    List, the strides of current Tensor.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1, 2, 3])
        >>> y = x[1]
        >>> print(y.get_strides())
        []
)DOC");                                  // NOLINT

static PyObject* tensor_method_strides(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  std::vector<int64_t> value;
  if (!self->tensor.defined() ||
      (!self->tensor.is_dense_tensor() && !self->tensor.is_dist_tensor())) {
    return ToPyObject(value);
  }
  auto stride = self->tensor.strides();
  int rank = static_cast<int>(stride.size());
  value.resize(rank);
  for (int i = 0; i < rank; i++) {
    value[i] = stride[i];
  }
  return ToPyObject(value);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_contiguous__doc__,  // NOLINT
             R"DOC(contiguous($self, /)
--

Returns a contiguous in memory tensor containing the same data as current Tensor.
If self tensor is already contiguous, this function returns the current Tensor.

Returns:
    Tensor, The contiguous Tensor.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1, 2, 3])
        >>> y = x[1]
        >>> y = y.contiguous()
        >>> print(y)
        Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True, 2)
)DOC");                                 // NOLINT

static PyObject* tensor_contiguous(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.is_dense_tensor() || self->tensor.is_dist_tensor()) {
    phi::DenseTensor* dense_tensor = nullptr;
    if (self->tensor.is_dist_tensor()) {
      dense_tensor =
          static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get())
              ->unsafe_mutable_value();
    } else {
      dense_tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
    }
    if (dense_tensor->meta().is_contiguous()) {
      Py_INCREF(self);
      return reinterpret_cast<PyObject*>(self);
    } else {
      eager_gil_scoped_release guard;
      EagerSetDeviceId();
      *dense_tensor = paddle::experimental::Trans2Contiguous(*dense_tensor);
      Py_INCREF(self);
      return reinterpret_cast<PyObject*>(self);
    }
  } else {
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_contiguous__doc__,  // NOLINT
             R"DOC(is_contiguous($self, /)
--

Whether the Tensor is contiguous.

Returns:
    Bool, Whether the Tensor is contiguous.

Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1, 2, 3])
        >>> y = x[1]
        >>> print(y.is_contiguous())
)DOC");                                    // NOLINT

static PyObject* tensor_is_contiguous(TensorObject* self,
                                      PyObject* args,
                                      PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.is_dense_tensor()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
    return ToPyObject(dense_tensor->meta().is_contiguous());
  } else if (self->tensor.is_dist_tensor()) {
    auto dense_tensor = std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                            self->tensor.impl())
                            ->unsafe_mutable_value();
    return ToPyObject(dense_tensor->meta().is_contiguous());
  } else {
    return ToPyObject(true);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_sparse_dim__doc__,
             R"DOC(sparse_dim($self, /)
--

Returns the number of sparse dimensions of sparse Tensor.

Note:
    **If self is not sparse Tensor, return 0.**

Returns:
    int, sparse dim of self Tensor

Examples:

    .. code-block:: python

        >>> import paddle

        >>> indices = [[0, 1, 2], [1, 2, 0]]
        >>> values = [1.0, 2.0, 3.0]
        >>> dense_shape = [3, 3]
        >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        >>> coo.sparse_dim()
        2

        >>> crows = [0, 2, 3, 5]
        >>> cols = [1, 3, 2, 0, 1]
        >>> values = [1, 2, 3, 4, 5]
        >>> dense_shape = [3, 4]
        >>> csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
        >>> csr.sparse_dim()
        2

        >>> dense = paddle.to_tensor([1, 2, 3])
        >>> dense.sparse_dim()
        0

)DOC");  // NOLINT

static PyObject* tensor_method_sparse_dim(TensorObject* self,
                                          PyObject* args,
                                          PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.is_sparse_coo_tensor()) {
    auto sparse_coo_tensor =
        std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
    return ToPyObject(sparse_coo_tensor->sparse_dim());
  } else if (self->tensor.is_sparse_csr_tensor()) {
    auto sparse_csr_tensor =
        std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
    return ToPyObject(sparse_csr_tensor->sparse_dim());
  } else {
    return ToPyObject(0);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_dense_dim__doc__,
             R"DOC(dense_dim($self, /)
--

Returns the number of dense dimensions of sparse Tensor.

Note:
    **If self is not sparse Tensor, return len(self.shape).**

Returns:
    int, dense dim of self Tensor

Examples:

    .. code-block:: python

        >>> import paddle
        >>> import numpy as np

        >>> indices = [[0, 1, 1], [2, 0, 2]]
        >>> values = np.array([[3, 4], [5, 6], [7, 8]])
        >>> dense_shape = [2, 3, 2]
        >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        >>> coo.dense_dim()
        1

        >>> crows = [0, 2, 3, 5]
        >>> cols = [1, 3, 2, 0, 1]
        >>> values = [1, 2, 3, 4, 5]
        >>> dense_shape = [3, 4]
        >>> csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
        >>> csr.dense_dim()
        0

        >>> dense = paddle.to_tensor([[1, 2, 3]])
        >>> dense.dense_dim()
        >>> 2

)DOC");  // NOLINT

static PyObject* tensor_method_dense_dim(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.is_sparse_coo_tensor()) {
    auto sparse_coo_tensor =
        std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
    return ToPyObject(sparse_coo_tensor->dense_dim());
  } else if (self->tensor.is_sparse_csr_tensor()) {
    auto sparse_csr_tensor =
        std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
    return ToPyObject(sparse_csr_tensor->dense_dim());
  } else {
    return ToPyObject(self->tensor.shape().size());
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__set_impl(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Running in tensor_method__set_impl: set Tensor impl form the "
             "other Tensor.";
  auto tensor = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  self->tensor.set_impl(tensor.impl());
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

#if defined(PADDLE_WITH_CUDA)
static PyObject* tensor_method__uva(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Running in tensor_method__uva.";
  PADDLE_ENFORCE_EQ(
      self->tensor.is_dense_tensor() || self->tensor.is_dist_tensor(),
      true,
      common::errors::InvalidArgument("Unified virtual addressing only support "
                                      "DenseTensor and DistTensor currently."));
  PADDLE_ENFORCE_EQ(
      phi::is_cpu_place(self->tensor.place()),
      true,
      common::errors::InvalidArgument("Unified virtual addressing only support "
                                      "CPU Tensor currently."));
  int device_id =
      pybind::CastPyArg2AttrLong(PyTuple_GET_ITEM(args, 0), 0);  // NOLINT
  phi::DenseTensor* dense_tensor = nullptr;
  if (self->tensor.is_dist_tensor()) {
    dense_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get())
            ->unsafe_mutable_value();
  } else {
    dense_tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  }
  tensor_uva(dense_tensor, device_id);

  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}
#endif
static PyObject* tensor_method__is_string_tensor_hold_allocation(
    TensorObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  auto string_tensor =
      std::dynamic_pointer_cast<phi::StringTensor>(self->tensor.impl());
  if (string_tensor) {
    return ToPyObject(string_tensor->initialized());
  } else {
    return ToPyObject(false);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef variable_methods[] = {  // NOLINT
    {"numpy",
     (PyCFunction)(void (*)())tensor_method_numpy,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_numpy__doc__},
    {"_is_initialized",
     (PyCFunction)(void (*)())tensor_method__is_initialized,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_dense_tensor_hold_allocation",
     (PyCFunction)(void (*)())tensor_method__is_dense_tensor_hold_allocation,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_copy_to",
     (PyCFunction)(void (*)())tensor_method__copy_to,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"copy_",
     (PyCFunction)(void (*)())tensor_method_copy_,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"clone",
     (PyCFunction)(void (*)())tensor_method_clone,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_clone__doc__},
    {"reconstruct_from_",
     (PyCFunction)(void (*)())tensor_method_reconstruct_from_,
     METH_VARARGS | METH_KEYWORDS,
     tensor_reconstruct_from___doc__},
    {"retain_grads",
     (PyCFunction)(void (*)())tensor_retain_grads,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_retain_grads__doc__},
    {"clear_gradient",
     (PyCFunction)(void (*)())tensor_clear_gradient,
     METH_VARARGS | METH_KEYWORDS,
     tensor_clear_gradient__doc__},
    {"is_dense",
     (PyCFunction)(void (*)())tensor_method_is_dense,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_is_dense__doc__},
    {"is_dist",
     (PyCFunction)(void (*)())tensor_method_is_dist,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_is_dist__doc__},
    {"_zero_grads",
     (PyCFunction)(void (*)())tensor__zero_grads,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_to_dist_",
     (PyCFunction)(void (*)())tensor__to_dist,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_share_buffer_to",
     (PyCFunction)(void (*)())tensor__share_buffer_to,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_unsafe_share_buffer_to",
     (PyCFunction)(void (*)())tensor__unsafe_share_buffer_to,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_shared_buffer_with",
     (PyCFunction)(void (*)())tensor__is_shared_buffer_with,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_share_underline_tensor_to",
     (PyCFunction)(void (*)())tensor__share_underline_tensor_to,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_shared_underline_tensor_with",
     (PyCFunction)(void (*)())tensor__is_shared_underline_tensor_with,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"detach",
     (PyCFunction)(void (*)())tensor_method_detach,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_detach__doc__},
    {"detach_",
     (PyCFunction)(void (*)())tensor_method_detach_,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_detach___doc__},
    {"get_tensor",
     (PyCFunction)(void (*)())tensor_method_get_underline_tensor,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_get_tensor__doc__},
    {"get_selected_rows",
     (PyCFunction)(void (*)())tensor_method_get_underline_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_get_tensor_from_selected_rows",
     (PyCFunction)(void (*)())tensor_method__get_tensor_from_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_getitem_dygraph",
     (PyCFunction)(void (*)())tensor__getitem_dygraph,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_getitem_from_offset",
     (PyCFunction)(void (*)())tensor__getitem_from_offset,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_setitem_dygraph",
     (PyCFunction)(void (*)())tensor__setitem_dygraph,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_apply",
     (PyCFunction)(void (*)())tensor_apply,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_apply_",
     (PyCFunction)(void (*)())tensor_apply_,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_register_grad_hook",
     (PyCFunction)(void (*)())tensor_register_grad_hook,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_inplace_assign",  // NOTE(xiongkun03): only used in sot.
     (PyCFunction)(void (*)())tensor_inplace_assign,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_remove_grad_hook",
     (PyCFunction)(void (*)())tensor_remove_grad_hook,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_register_backward_hook",
     (PyCFunction)(void (*)())tensor_register_reduce_hook,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method__register_reduce_hook__doc__},
    {"_set_grad_type",
     (PyCFunction)(void (*)())tensor__set_grad_type,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_clear",
     (PyCFunction)(void (*)())tensor__clear,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_clear_dataptr",
     (PyCFunction)(void (*)())tensor__clear_dataptr,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_copy_gradient_from",
     (PyCFunction)(void (*)())tensor__copy_gradient_from,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_tensor_use_gpudnn",
     (PyCFunction)(void (*)())tensor__use_gpudnn,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    /** the methods to adapt old dygraph, will be removed in the future **/
    {"set_string_list",
     (PyCFunction)(void (*)())tensor_method_set_string_list,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"set_vocab",
     (PyCFunction)(void (*)())tensor_method_set_vocab,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"get_map_tensor",
     (PyCFunction)(void (*)())tensor_method_get_map_tensor,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    /***the method of sparse tensor****/
    {"nnz",
     (PyCFunction)(void (*)())tensor_method_get_non_zero_nums,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_nnz__doc__},
    {"indices",
     (PyCFunction)(void (*)())tensor_method_get_non_zero_indices,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_indices__doc__},
    {"values",
     (PyCFunction)(void (*)())tensor_method_get_non_zero_elements,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_values__doc__},
    {"crows",
     (PyCFunction)(void (*)())tensor_method_get_non_zero_crows,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_crows__doc__},
    {"cols",
     (PyCFunction)(void (*)())tensor_method_get_non_zero_cols,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_cols__doc__},
    {"is_sparse",
     (PyCFunction)(void (*)())tensor_method_is_sparse,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_sparse__doc__},
    {"is_sparse_coo",
     (PyCFunction)(void (*)())tensor_method_is_sparse_coo,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_sparse_coo__doc__},
    {"is_sparse_csr",
     (PyCFunction)(void (*)())tensor_method_is_sparse_csr,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_sparse_csr__doc__},
    {"is_same_shape",
     (PyCFunction)(void (*)())tensor_method_is_same_shape,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_same_shape__doc__},
    {"to_sparse_csr",
     (PyCFunction)(void (*)())tensor_method_to_sparse_csr,
     METH_VARARGS | METH_KEYWORDS,
     tensor_to_sparse_csr__doc__},
    {"is_coalesced",
     (PyCFunction)(void (*)())tensor_method_is_coalesced,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_coalesced__doc__},
    {"sparse_dim",
     (PyCFunction)(void (*)())tensor_method_sparse_dim,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_sparse_dim__doc__},
    {"dense_dim",
     (PyCFunction)(void (*)())tensor_method_dense_dim,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_dense_dim__doc__},
    /***the method of sparse tensor****/
    {"element_size",
     (PyCFunction)(void (*)())tensor_method_element_size,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_element_size__doc__},
    {"_inplace_version",
     (PyCFunction)(void (*)())tensor__inplace_version,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_bump_inplace_version",
     (PyCFunction)(void (*)())tensor__bump_inplace_version,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method__bump_inplace_version__doc__},
    {"is_selected_rows",
     (PyCFunction)(void (*)())tensor_method_is_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"rows",
     (PyCFunction)(void (*)())tensor_method_get_rows,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_reset_grad_inplace_version",
     (PyCFunction)(void (*)())tensor__reset_grad_inplace_version,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_share_memory",
     (PyCFunction)(void (*)())tensor_method__share_memory,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_offset",
     (PyCFunction)(void (*)())tensor__offset,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_grad_name",
     (PyCFunction)(void (*)())tensor__grad_name,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_grad_value",
     (PyCFunction)(void (*)())tensor__grad_value,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_local_value",
     (PyCFunction)(void (*)())tensor__local_value,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_unset_fake_empty",
     (PyCFunction)(void (*)())tensor__unset_fake_empty,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"data_ptr",
     (PyCFunction)(void (*)())tensor_data_ptr,
     METH_VARARGS | METH_KEYWORDS,
     tensor_data_ptr__doc__},
    {"_grad_ivar",
     (PyCFunction)(void (*)())tensor__grad_ivar,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"contiguous",
     (PyCFunction)(void (*)())tensor_contiguous,
     METH_VARARGS | METH_KEYWORDS,
     tensor_contiguous__doc__},
    {"is_contiguous",
     (PyCFunction)(void (*)())tensor_is_contiguous,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_contiguous__doc__},
    {"get_strides",
     (PyCFunction)(void (*)())tensor_method_strides,
     METH_VARARGS | METH_KEYWORDS,
     tensor_get_strides__doc__},
    {"_set_impl",
     (PyCFunction)(void (*)())tensor_method__set_impl,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
#if defined(PADDLE_WITH_CUDA)
    {"_tensor_uva",
     (PyCFunction)(void (*)())tensor_method__uva,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
#endif
    {nullptr, nullptr, 0, nullptr}};

// variable_methods for core.eager.StringTensor
PyMethodDef string_tensor_variable_methods[] = {  // NOLINT
    {"numpy",
     (PyCFunction)(void (*)())tensor_method_numpy_for_string_tensor,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_initialized",
     (PyCFunction)(void (*)())tensor_method__is_initialized,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_string_tensor_hold_allocation",
     (PyCFunction)(void (*)())tensor_method__is_string_tensor_hold_allocation,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    // TODO(zhoushunjie): Need to add _copy_to, copy_ for StringTensor.
    {nullptr, nullptr, 0, nullptr}};
}  // namespace paddle::pybind
