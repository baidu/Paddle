// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Python.h>
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include <algorithm>
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/scope_guard.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/funcs/strided_slice.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

using egr::ConvertAllInputsToDistTensor;
using egr::InputsContainDistTensor;

namespace py = pybind11;

namespace paddle {
namespace pybind {

template <typename T>
inline T GetDenseTensorValue(const phi::DenseTensor* x) {
  T value = static_cast<T>(0);
  if (!(x->place().GetType() == phi::AllocationType::CPU)) {
    phi::DenseTensor cpu_x;
    framework::TensorCopy(*x, phi::CPUPlace(), &cpu_x);
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    const phi::DeviceContext* dev_ctx = pool.Get(x->place());
    dev_ctx->Wait();
#endif
    value = cpu_x.data<T>()[0];
  } else {
    value = x->data<T>()[0];
  }
  return value;
}

static Py_ssize_t GetSliceIndexFromPyObject(PyObject* obj);
// Slice related methods
static bool PyCheckInteger(PyObject* obj) {
#if PY_VERSION_HEX < 0x03000000
  return (PyLong_Check(obj) || PyInt_Check(obj)) && !PyBool_Check(obj);
#else
  return PyLong_Check(obj) && !PyBool_Check(obj);
#endif
}

static bool IsNumpyType(PyObject* obj) {
  // It is not a good way to judge the type of obj by its type'name. Maybe using
  // `PyArray_IsScalar` will be better. However, this interface cannot be used
  // by including pybind11, and it needs to compile with numpy.
  auto type_name = std::string(Py_TYPE(obj)->tp_name);
  return type_name == "numpy.int64" || type_name == "numpy.longlong" ||
         type_name == "numpy.int32" || type_name == "numpy.int16";
}

static Py_ssize_t GetSliceIndexFromTensor(const phi::DenseTensor& tensor) {
  if (tensor.numel() == 1) {
    if (framework::TransToProtoVarType(tensor.type()) ==
        framework::proto::VarType::INT32) {
      return static_cast<Py_ssize_t>(GetDenseTensorValue<int32_t>(&tensor));
    } else if (framework::TransToProtoVarType(tensor.type()) ==
               framework::proto::VarType::INT64) {
      return static_cast<Py_ssize_t>(GetDenseTensorValue<int64_t>(&tensor));
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Currently, the type of tensor in slice indices only allows "
          "int32 and int64, please check the type of index tensor."));
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, tensor in slice indices only allows 1 element, "
        "but received %d.",
        tensor.numel()));
  }
}

// NOTE(zhiqiu): Revised version of PySlice_GetIndices. From:
// https://github.com/python/cpython/blob/8d21aa21f2cbc6d50aab3f420bb23be1d081dac4/Objects/sliceobject.c#L103
// Original PySlice_GetIndices return wrong result when
// slice_item contains long int, such as arr[:180L].
// NOT sure why this happens !!!
// Besides, PySlice_GetIndices cannot raise error when float in slice item.
// So, I make a revised version of PySlice_GetIndices, named to
// _PySlice_GetIndices. Try to use _PySlice_Unpack which is more robust than
// PySlice_GetIndices in the future.
static int _PySlice_GetIndices(PySliceObject* r,
                               Py_ssize_t length,
                               Py_ssize_t* start,
                               Py_ssize_t* stop,
                               Py_ssize_t* step) {
  /* XXX support long ints */
  if (r->step == Py_None) {
    *step = 1;
  } else {
    if (PyCheckInteger(r->step) || IsNumpyType(r->step)) {
      *step = PyLong_AsLong(r->step);
    } else if (PyCheckTensor(r->step)) {
      *step = GetSliceIndexFromPyObject(r->step);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Currently, slice indices only allows None, integers, "
          "tensor(int) and numpy(int) in slice item, but received %s.",
          std::string(Py_TYPE(r->step)->tp_name)));
    }
  }
  if (r->start == Py_None) {
    *start = *step < 0 ? length - 1 : 0;
  } else {
    if (PyCheckInteger(r->start) || IsNumpyType(r->start)) {
      *start = PyLong_AsLong(r->start);
    } else if (PyCheckTensor(r->start)) {
      *start = GetSliceIndexFromPyObject(r->start);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Currently, slice indices only allows None, integers, "
          "tensor(int) and numpy(int) in slice item, but received %s.",
          std::string(Py_TYPE(r->start)->tp_name)));
    }
  }
  if (r->stop == Py_None) {
    *stop = *step < 0 ? -length - 1 : length;
  } else {
    if (PyCheckInteger(r->stop) || IsNumpyType(r->stop)) {
      *stop = PyLong_AsLong(r->stop);
    } else if (PyCheckTensor(r->stop)) {
      *stop = GetSliceIndexFromPyObject(r->stop);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Currently, slice indices only allows None, integers, "
          "tensor(int) and numpy(int) in slice item, but received %s.",
          std::string(Py_TYPE(r->stop)->tp_name)));
    }
  }

  // normalize start and stop
  bool dummy_zero_dim_out = false;
  phi::funcs::normalize_interval(
      *start, *stop, *step, length, start, stop, &dummy_zero_dim_out);
  // return value below seems to be useless...
  if (*stop > length) return -1;
  if (*start >= length) return -1;
  if (*step == 0) return -1;
  return 0;
}

static void ParseIndex(const paddle::Tensor& tensor,
                       PyObject* _index,
                       std::vector<int64_t>* slice_axes,
                       std::vector<int>* slice_starts,
                       std::vector<int>* slice_ends,
                       std::vector<int>* slice_strides,
                       std::vector<int64_t>* decrease_axis,
                       std::vector<int64_t>* none_axes,
                       std::vector<int64_t>* infer_flags,
                       std::vector<int>* advanced_index_dim,
                       std::vector<paddle::Tensor>* advanced_index,
                       bool* has_advanced_index,
                       bool* use_strided_slice) {
  // NOTE(zhiqiu): PyTuple_Pack increases refcount.
  PyObject* index = !PyTuple_Check(_index) ? PyTuple_Pack(1, _index) : _index;
  DEFINE_PADDLE_SCOPE_GUARD([index, _index]() {
    if (!PyTuple_Check(_index)) {
      Py_DECREF(index);
      VLOG(4) << "Call Py_DECREF";
    }
  });
  // for case 0-size tensor in slice
  PADDLE_ENFORCE_EQ(
      tensor.defined(),
      true,
      common::errors::InvalidArgument("tensor has not been defined"));
  const auto& shape = tensor.dims();
  const int rank = shape.size();
  const int size = PyTuple_GET_SIZE(index);

  // Check Ellipsis is valid
  int specified_dims = 0;
  int ell_count = 0;
  for (int dim = 0; dim < size; ++dim) {
    PyObject* slice_item = PyTuple_GetItem(index, dim);
    if (slice_item == Py_Ellipsis) {
      ell_count++;
    } else if (slice_item != Py_None && !PyBool_Check(slice_item)) {
      specified_dims++;
    }
  }
  PADDLE_ENFORCE_LE(ell_count,
                    1,
                    common::errors::InvalidArgument(
                        "An index can only have a single ellipsis ('...')"));

  // deal with indexing_item
  int none_count = 0;
  for (int i = 0, current_dim = 0, estimated_dim = 0; i < size; ++i) {
    PyObject* slice_item = PyTuple_GetItem(index, i);

    infer_flags->push_back(1);
    int64_t dim_len = shape[current_dim];
    if (PyCheckInteger(slice_item) || IsNumpyType(slice_item)) {
      // integer, PyLong_AsLong supports both int and long
      int64_t start = static_cast<int64_t>(PyLong_AsLong(slice_item));
      auto s_t = start;
      start = start < 0 ? start + dim_len : start;

      PADDLE_ENFORCE(
          0 <= start && start < dim_len,
          common::errors::OutOfRange("The starting index %d of slice is out "
                                     "of bounds in tensor %d-th axis, it "
                                     "shound be in the range of [%d, %d).",
                                     s_t,
                                     current_dim,
                                     -dim_len,
                                     dim_len));

      slice_axes->push_back(current_dim);
      slice_starts->push_back(start);
      slice_ends->push_back(start + 1);
      slice_strides->push_back(1);
      decrease_axis->push_back(current_dim);
      current_dim++;
    } else if (PySlice_Check(slice_item)) {
      // slice item
      Py_ssize_t start, end, step;
      PySliceObject* p = reinterpret_cast<PySliceObject*>(slice_item);
      _PySlice_GetIndices(p, dim_len, &start, &end, &step);

      // :: or : or 0:dim_len:1
      if (start == 0 && end == dim_len && step == 1) {
        current_dim++;
        estimated_dim++;
        continue;
      }
      slice_axes->push_back(current_dim);
      slice_starts->push_back(start);
      slice_ends->push_back(end);
      slice_strides->push_back(step);
      estimated_dim++;
      current_dim++;

      if (step != 1) {
        *use_strided_slice = true;
      }
    } else if (slice_item == Py_Ellipsis) {
      current_dim += rank - specified_dims;
      estimated_dim += rank - specified_dims;
    } else if (slice_item == Py_None) {
      none_axes->push_back(current_dim + none_count);
      none_count++;
    } else if (PyBool_Check(slice_item)) {
      *has_advanced_index = true;
      none_axes->push_back(current_dim + none_count);
      none_count++;
      bool index_ele = (slice_item == Py_True);
      auto slice_tensor =
          full_ad_func({1}, index_ele, phi::DataType::BOOL, tensor.place());
      advanced_index->push_back(std::move(slice_tensor));
      (*advanced_index_dim)[estimated_dim] = estimated_dim;
      estimated_dim++;
    } else if (PyCheckTensor(slice_item)) {
      auto slice_tensor = CastPyArg2Tensor(slice_item, 0);
      if (slice_tensor.shape().size() == 0) {
        if (slice_tensor.dtype() != phi::DataType::BOOL) {
          // 0-D int tensor is same with scalar
          PADDLE_ENFORCE_EQ(
              slice_tensor.is_dense_tensor(),
              true,
              common::errors::InvalidArgument(
                  "Now, Tensor in indexing only support DenseTensor."));
          Py_ssize_t s_t = GetSliceIndexFromTensor(
              (*static_cast<phi::DenseTensor*>(slice_tensor.impl().get())));
          auto start = s_t < 0 ? s_t + dim_len : s_t;

          PADDLE_ENFORCE(0 <= start && start < dim_len,
                         common::errors::OutOfRange(
                             "The starting index %d of slice is out "
                             "of bounds in tensor %d-th axis, it "
                             "shound be in the range of [%d, %d).",
                             s_t,
                             current_dim,
                             -dim_len,
                             dim_len));

          slice_axes->push_back(current_dim);
          slice_starts->push_back(start);
          slice_ends->push_back(start + 1);
          slice_strides->push_back(1);
          decrease_axis->push_back(current_dim);
          current_dim++;
        } else {
          // 0-D bool Tensor, same as single PY-bool.
          *has_advanced_index = true;
          none_axes->push_back(current_dim + none_count);
          none_count++;
          slice_tensor = unsqueeze_ad_func(slice_tensor, {-1});
          advanced_index->push_back(std::move(slice_tensor));
          (*advanced_index_dim)[estimated_dim] = estimated_dim;
          estimated_dim++;
        }
      } else {
        if (slice_tensor.dtype() == phi::DataType::BOOL) {
          PADDLE_ENFORCE_EQ(slice_tensor.shape()[0],
                            dim_len,
                            common::errors::OutOfRange(
                                "The shape of boolean index %d did not match"
                                "indexed tensor %d along axis %d.",
                                slice_tensor.shape()[0],
                                dim_len,
                                current_dim));
        }
        *has_advanced_index = true;
        advanced_index->push_back(std::move(slice_tensor));
        (*advanced_index_dim)[estimated_dim] = estimated_dim;
        estimated_dim++;
        current_dim++;
      }

    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Currently, Tensor.__indices__() only allows indexing "
          "by Boolean, Integers, Slices, Ellipsis, None, Tuples of these types "
          "and List / Tensor of Bool and Integers, but received "
          "%s in %dth slice item",
          std::string(Py_TYPE(slice_item)->tp_name),
          i + 1));
    }
  }

  // valid_index is the number of dimensions exclude None index
  const int valid_indices = size - none_axes->size() - ell_count;
  PADDLE_ENFORCE_EQ(valid_indices <= rank,
                    true,
                    common::errors::InvalidArgument(
                        "Too many indices (%d) for tensor of dimension %d.",
                        valid_indices,
                        rank));
}

static paddle::Tensor getTensorWithBasicIndexing(
    const paddle::Tensor& tensor,
    std::vector<int64_t>* slice_axes,
    std::vector<int>* slice_starts,
    std::vector<int>* slice_ends,
    std::vector<int>* slice_strides,
    std::vector<int64_t>* decrease_axis,
    std::vector<int64_t>* none_axes,
    std::vector<int64_t>* infer_flags,
    bool* use_strided_slice,
    bool* out_is_view) {
  paddle::Tensor out;
  if (slice_axes->empty()) {
    out = tensor;
  } else {
    *out_is_view = true;
    if (!(*use_strided_slice)) {
      eager_gil_scoped_release guard;
      out = slice_ad_func(tensor,
                          *slice_axes,
                          *slice_starts,
                          *slice_ends,
                          *infer_flags,
                          *decrease_axis);
    } else {
      eager_gil_scoped_release guard;
      std::vector<int> slice_axes_int32(slice_axes->begin(), slice_axes->end());

      out = strided_slice_ad_func(
          tensor, slice_axes_int32, *slice_starts, *slice_ends, *slice_strides);
      if (!decrease_axis->empty()) {
        out = squeeze_ad_func(out, *decrease_axis);
      }
    }
  }
  if (!none_axes->empty()) {
    *out_is_view = true;
    eager_gil_scoped_release guard;
    // Deal with cases that decrease_axes is not empty
    // For example:
    // # x.shape: (2,3,4)
    // out = x[0, 0:2, None] # out.shape : (2, 1, 4)
    for (auto& axis : *(none_axes)) {
      int len = 0;
      for (int64_t da : *decrease_axis) {
        if (da < axis) {
          len++;
        }
      }
      axis -= len;
    }
    out = unsqueeze_ad_func(out, *none_axes);
  }
  return out;
}

static paddle::Tensor dealWithAdvancedIndex(
    const paddle::Tensor& tensor,
    std::vector<int>* advanced_index_dim,
    std::vector<paddle::Tensor>* advanced_index,
    bool is_for_setitem,
    std::vector<paddle::Tensor>* transed_index,
    std::vector<int>* trans_back_dim,
    int* pos_of_new_dim,
    int* rank_of_new_dim,
    std::vector<int>* trans_dim,
    bool* out_is_view) {
  int p = 0;
  for (size_t i = 0; i < advanced_index_dim->size(); ++i) {
    auto index_dim = (*advanced_index_dim)[i];
    if (index_dim != -1) {
      // size of advanced_index is same to number of non -1 element in
      // advanced_index_dim
      auto index = (*advanced_index)[p++];

      if (index_dim == 0) {
        // case 1: advanced indices at axis 0, the new dim will be at first.
        *pos_of_new_dim = 0;
      } else if (index_dim > 0 && trans_dim->size() > 0 &&
                 (*trans_dim)[trans_dim->size() - 1] != index_dim - 1) {
        // case 2: there are not adjacent advanced indices, the new dim will
        // be at first.
        *pos_of_new_dim = 0;
      } else {
        *pos_of_new_dim = std::min(index_dim, *pos_of_new_dim);
      }
      *rank_of_new_dim =
          std::max(*rank_of_new_dim, static_cast<int>(index.shape().size()));

      trans_dim->push_back(index_dim);
      transed_index->push_back(std::move(index));
    }
  }

  for (size_t i = 0; i < tensor.shape().size(); ++i) {
    if ((*advanced_index_dim)[i] == -1) {
      trans_dim->push_back(i);
    }
  }

  paddle::Tensor transed_tensor;

  // skip transform if the `trans_dim` is original order.
  std::vector<int> original_dim_order(tensor.shape().size());
  std::iota(original_dim_order.begin(), original_dim_order.end(), 0);

  if (original_dim_order == *trans_dim) {
    transed_tensor = tensor;
  } else {
    *out_is_view = true;
    transed_tensor = transpose_ad_func(tensor, *trans_dim);
  }

  if (is_for_setitem) {
    trans_back_dim->resize(trans_dim->size());
    std::iota(trans_back_dim->begin(), trans_back_dim->end(), 0);
    std::sort(trans_back_dim->begin(),
              trans_back_dim->end(),
              [&trans_dim](int left, int right) {
                return (*trans_dim)[left] < (*trans_dim)[right];
              });
  }
  return transed_tensor;
}

static paddle::Tensor getValueForBoolTensor(const paddle::Tensor& tensor,
                                            const paddle::Tensor& bool_index) {
  PADDLE_ENFORCE(bool_index.shape().size() <= tensor.shape().size(),
                 common::errors::InvalidArgument(
                     "The dims of bool index doesn't match indexed array, "
                     "the dims of bool index except to be equal or less "
                     "than %d, but received %d}.",
                     tensor.shape().size(),
                     bool_index.shape().size()));
  auto tensor_shape = tensor.shape();
  size_t i = 0;
  while (i < bool_index.shape().size()) {
    PADDLE_ENFORCE_EQ(
        bool_index.shape()[i],
        tensor_shape[i],
        common::errors::OutOfRange(
            "The dimension of bool index doesn't match indexed array along "
            "dimension %d, the target dimension is %d, but received %d",
            i,
            tensor_shape[i],
            bool_index.shape()[i]));
    i++;
  }

  const phi::distributed::ProcessMesh* mesh = nullptr;
  if (InputsContainDistTensor(&mesh, tensor, bool_index)) {
    ConvertAllInputsToDistTensor(mesh, tensor, bool_index);
  }

  if (bool_index.shape().size() == tensor_shape.size()) {
    return masked_select_ad_func(tensor, bool_index);
  }
  auto bool_2_idx = nonzero_ad_func(bool_index);
  return gather_nd_ad_func(tensor, bool_2_idx);
}

static void ParseBoolAndBroadcastIndices(
    std::vector<paddle::Tensor>* advanced_index) {
  for (size_t i = 0; i < advanced_index->size(); i++) {
    if ((*advanced_index)[i].dtype() == phi::DataType::BOOL) {
      paddle::Tensor bool_2_idx = nonzero_ad_func((*advanced_index)[i]);
      paddle::Tensor bool_2_idx_sliced =
          slice_ad_func(bool_2_idx, {1}, {0}, {1}, {1}, {1});
      (*advanced_index)[i] = bool_2_idx_sliced;
    }
  }
  if (advanced_index->size() > 1) {
    bool need_broadcast = false;
    common::DDim common_shape = common::make_ddim((*advanced_index)[0].shape());
    for (size_t i = 1; i < advanced_index->size(); ++i) {
      common::DDim current_shape =
          common::make_ddim((*advanced_index)[i].shape());
      if (current_shape != common_shape) {
        need_broadcast = true;
        common_shape =
            phi::funcs::BroadcastTwoDims(current_shape, common_shape, -1);
      }
    }

    if (need_broadcast) {
      // Here advanced_index has been checked ContainDistTensor
      // and transed in dealWithAdvancedIndex
      auto common_shape_vec = common::vectorize<int64_t>(common_shape);
      for (size_t i = 0; i < advanced_index->size(); ++i) {
        auto current_shape = (*advanced_index)[i].shape();
        if (current_shape != common_shape_vec) {
          (*advanced_index)[i] =
              expand_ad_func((*advanced_index)[i], common_shape_vec);
        }
      }
    }
  }
}

static paddle::Tensor dealWithValues(const paddle::Tensor& tensor,
                                     PyObject* value_obj,
                                     std::vector<phi::Scalar>* values,
                                     const bool trans_to_tensor) {
  paddle::Tensor value_tensor;
  if (PyCheckTensor(value_obj)) {
    value_tensor = reinterpret_cast<TensorObject*>(value_obj)->tensor;
  } else if (py::isinstance<py::array>(value_obj)) {
    paddle::Tensor value_tensor_tmp(
        std::make_shared<phi::DenseTensor>(),
        egr::Controller::Instance().GenerateUniqueName());
    py::object value_obj_tmp = py::reinterpret_borrow<py::object>(value_obj);
    py::object value = value_obj_tmp;
    if (tensor.dtype() == phi::DataType::FLOAT32) {
      if (!py::isinstance<py::array_t<float>>(value_obj_tmp)) {
        value = pybind11::detail::CastNumpyArray<float>(value_obj_tmp);
      }
    } else if (tensor.dtype() == phi::DataType::FLOAT64) {
      if (!py::isinstance<py::array_t<double>>(value_obj_tmp)) {
        value = pybind11::detail::CastNumpyArray<double>(value_obj_tmp);
      }
    } else if (tensor.dtype() == phi::DataType::INT32) {
      if (!py::isinstance<py::array_t<int32_t>>(value_obj_tmp)) {
        value = pybind11::detail::CastNumpyArray<int32_t>(value_obj_tmp);
      }
    } else if (tensor.dtype() == phi::DataType::INT64) {
      if (!py::isinstance<py::array_t<int64_t>>(value_obj_tmp)) {
        value = pybind11::detail::CastNumpyArray<int64_t>(value_obj_tmp);
      }
    } else if (tensor.dtype() == phi::DataType::BOOL) {
      if (!py::isinstance<py::array_t<bool>>(value_obj_tmp)) {
        value = pybind11::detail::CastNumpyArray<bool>(value_obj_tmp);
      }
    } else if (tensor.dtype() == phi::DataType::COMPLEX64) {
      if (!py::isinstance<py::array_t<std::complex<float>>>(value_obj_tmp)) {
        value = pybind11::detail::CastNumpyArray<std::complex<float>>(
            value_obj_tmp);
      }
    } else if (tensor.dtype() == phi::DataType::COMPLEX128) {
      if (!py::isinstance<py::array_t<std::complex<double>>>(value_obj_tmp)) {
        value = pybind11::detail::CastNumpyArray<std::complex<double>>(
            value_obj_tmp);
      }
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "When assign a numpy.np value to a paddle.Tensor, "
          "the data type of the paddle.Tensor must be bool, "
          "float32, float64, complex64, complex128, int32 or int64, "
          "please check the type of tensor."));
    }
    SetTensorFromPyArray(
        static_cast<phi::DenseTensor*>(value_tensor_tmp.impl().get()),
        value,
        tensor.place(),
        false);
    value_tensor = value_tensor_tmp;
  } else {
    py::object value_obj_tmp = py::reinterpret_borrow<py::object>(value_obj);
    // convert the value to self data type
    if (py::isinstance<py::float_>(value_obj_tmp) ||
        py::isinstance<py::int_>(value_obj_tmp) ||
        py::isinstance<py::bool_>(value_obj_tmp) ||
        PyComplex_Check(value_obj)) {
      if (tensor.dtype() == phi::DataType::FLOAT32 ||
          tensor.dtype() == phi::DataType::FLOAT16 ||
          tensor.dtype() == phi::DataType::BFLOAT16) {
        values->push_back(value_obj_tmp.cast<float>());
      } else if (tensor.dtype() == phi::DataType::FLOAT64) {
        values->push_back(value_obj_tmp.cast<double>());
      } else if (tensor.dtype() == phi::DataType::INT32 ||
                 tensor.dtype() == phi::DataType::INT16 ||
                 tensor.dtype() == phi::DataType::INT8 ||
                 tensor.dtype() == phi::DataType::UINT8) {
        values->push_back(value_obj_tmp.cast<float>());
      } else if (tensor.dtype() == phi::DataType::INT64) {
        values->push_back(value_obj_tmp.cast<double>());
      } else if (tensor.dtype() == phi::DataType::BOOL) {
        values->push_back(value_obj_tmp.cast<bool>());
      } else if (tensor.dtype() == phi::DataType::COMPLEX64) {
        values->push_back(value_obj_tmp.cast<std::complex<float>>());
      } else if (tensor.dtype() == phi::DataType::COMPLEX128) {
        values->push_back(value_obj_tmp.cast<std::complex<double>>());
      }
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Value type error. The assign value allows "
          "Tensor, numpy.ndarray, integer, float, complex or bool, "
          "but received %s.",
          Py_TYPE(value_obj)));
    }

    if (trans_to_tensor) {
      value_tensor =
          full_ad_func({1}, (*values)[0], tensor.dtype(), tensor.place());
    }
  }
  return value_tensor;
}

}  // namespace pybind
}  // namespace paddle
