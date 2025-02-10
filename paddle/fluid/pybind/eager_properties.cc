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
#include <Python.h>
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include <string>
#include <vector>

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/memcpy.h"

using egr::ConvertAllInputsToDistTensor;
using egr::InputsContainDistTensor;

#pragma GCC diagnostic ignored "-Wwrite-strings"

COMMON_DECLARE_bool(enable_pir_api);

namespace paddle::pybind {

extern PyTypeObject* p_tensor_type;

PyDoc_STRVAR(tensor_name__doc__,  // NOLINT
             R"DOC(name

Tensor's name.

Returns:
    str: Tensor's name.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor(1.)
        >>> print(x.name)
        generated_tensor_0
        >>> x.name = 'test_tensor_name'
        >>> print(x.name)
        test_tensor_name
)DOC");

PyObject* tensor_properties_get_name(TensorObject* self, void* closure) {
  EAGER_TRY
  // NOTE(dev): [why not use egr::Controller::Instance::GenerateUniqueName()?]
  // Because Controller must holder a tracer, but 'tensor.name' maybe called
  // everywhere such as static graph mode in @to_static, which means tracer is
  // None.
  static egr::UniqueNameGenerator name_generator;
  if (self->tensor.name().empty()) {
    self->tensor.set_name(name_generator.Generate());
  }
  return ToPyObject(self->tensor.name());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_type__doc__,  // NOLINT
             R"DOC(type

Tensor's type.

Returns:
    VarType: Tensor's type.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor(1.)
        >>> print(x.type)
        VarType.DENSE_TENSOR
)DOC");

PyObject* tensor_properties_get_type(TensorObject* self, void* closure) {
  EAGER_TRY
  if (!self->tensor.defined() || self->tensor.is_dense_tensor() ||
      self->tensor.is_dist_tensor()) {
    // be same to old dygraph
    return ToPyObject(paddle::framework::proto::VarType::DENSE_TENSOR);
  }
  if (self->tensor.is_selected_rows()) {
    return ToPyObject(paddle::framework::proto::VarType::SELECTED_ROWS);
  } else if (egr::IsVariableCompatTensor(self->tensor)) {
    return ToPyObject(static_cast<paddle::framework::proto::VarType::Type>(
        static_cast<const egr::VariableCompatTensor*>(self->tensor.impl().get())
            ->Type()));
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_leaf__doc__,  // NOLINT
             R"DOC(is_leaf

Whether a Tensor is leaf Tensor.

For the Tensor whose stop_gradient is ``True`` , it will be leaf Tensor.

For the Tensor whose stop_gradient is ``False`` , it will be leaf Tensor too if it is created by user.

Returns:
    bool: Whether a Tensor is leaf Tensor.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor(1.)
        >>> print(x.is_leaf)
        True

        >>> x = paddle.to_tensor(1., stop_gradient=True)
        >>> y = x + 1
        >>> print(x.is_leaf)
        True

        >>> print(y.is_leaf)
        True

        >>> x = paddle.to_tensor(1., stop_gradient=False)
        >>> y = x + 1
        >>> print(x.is_leaf)
        True

        >>> print(y.is_leaf)
        False
)DOC");

PyObject* tensor_properties_is_leaf(TensorObject* self, void* closure) {
  EAGER_TRY
  return ToPyObject(egr::EagerUtils::IsLeafTensor(self->tensor));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_name(TensorObject* self,
                               PyObject* value,
                               void* closure) {
  EAGER_TRY
  self->tensor.set_name(CastPyArg2AttrString(value, 0));
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

PyDoc_STRVAR(tensor_stop_gradient__doc__,  // NOLINT
             R"DOC(stop_gradient

Tensor's stop_gradient.

Returns:
    bool: Tensor's stop_gradient.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor(1.)
        >>> print(x.stop_gradient)
        True

        >>> x.stop_gradient = False
        >>> print(x.stop_gradient)
        False
)DOC");

PyObject* tensor_properties_get_stop_gradient(TensorObject* self,
                                              void* closure) {
  EAGER_TRY
  auto meta = egr::EagerUtils::autograd_meta(&self->tensor);
  return ToPyObject(meta->StopGradient());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_data__doc__,  // NOLINT
             R"DOC(data

Tensor's self.

Returns:
    Tensor: self.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor(1.)
        >>> print(x)
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
        1.)

        >>> print(x.data)
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
        1.)

        >>> x.data = paddle.to_tensor(2.)
        >>> print(x)
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
        2.)

        >>> print(x.data)
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
        2.)
)DOC");
PyObject* tensor_properties_get_data(TensorObject* self, void* closure) {
  EAGER_TRY
  Py_INCREF(self);
  return reinterpret_cast<PyObject*>(self);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_data(TensorObject* self,
                               PyObject* value,
                               void* closure) {
  EAGER_TRY
  auto src = CastPyArg2Tensor(value, 0);
  self->tensor = src;
  phi::DenseTensor tmp;
  if (self->tensor.is_dense_tensor()) {
    auto dense_tensor =
        static_cast<phi::DenseTensor*>(self->tensor.impl().get());
    if (dense_tensor) {
      dense_tensor->ShareInplaceVersionCounterWith(tmp);
    }
  } else if (self->tensor.is_dist_tensor()) {
    auto dist_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get())
            ->unsafe_mutable_value();
    if (dist_tensor) {
      dist_tensor->ShareInplaceVersionCounterWith(tmp);
    }
  }
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

PyDoc_STRVAR(tensor_grad__doc__,  // NOLINT
             R"DOC(grad

Tensor's grad Tensor.

Returns:
    Tensor: grad Tensor.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor(1.0, stop_gradient=False)
        >>> y = x**2
        >>> y.backward()
        >>> print(x.grad)
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,
        2.)

        >>> x.grad = paddle.to_tensor(3.0)
        >>> print(x.grad)
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,
        3.)
)DOC");
PyObject* tensor_properties_get_grad(TensorObject* self, void* closure) {
  EAGER_TRY
  VLOG(6) << "Get grad for tensor: " << self->tensor.name();
  auto meta = egr::EagerUtils::nullable_autograd_meta(self->tensor);
  if (meta && meta->Grad().has_allocation()) {
    return ToPyObject(meta->Grad());
  } else {
    if (meta && !meta->Grad().has_allocation() && meta->Grad().impl() &&
        meta->Grad().is_dist_tensor()) {
      return ToPyObject(meta->Grad(), false);
    }
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_grad(TensorObject* self,
                               PyObject* value,
                               void* closure) {
  EAGER_TRY
  auto& src = CastPyArg2Tensor(value, 0);
  PADDLE_ENFORCE(egr::EagerUtils::IsLeafTensor(self->tensor),
                 common::errors::Fatal("Only leaf Tensor can be set grad."));

  paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE(
      grad != nullptr,
      common::errors::Fatal("Detected NULL grad"
                            "Please check if you have manually cleared"
                            "the grad inside autograd_meta"));
  const phi::distributed::ProcessMesh* mesh = nullptr;
  if (InputsContainDistTensor(&mesh, src, self->tensor, *grad)) {
    ConvertAllInputsToDistTensor(mesh, src, self->tensor, *grad);
  }
  grad->copy_(src, self->tensor.place(), true);
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

int tensor_properties_set_grad_(TensorObject* self,
                                PyObject* value,
                                void* closure) {
  EAGER_TRY
  auto src = CastPyArg2Tensor(value, 0);
  PADDLE_ENFORCE(egr::EagerUtils::IsLeafTensor(self->tensor),
                 common::errors::Fatal("Only leaf Tensor can be set grad."));

  paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE(
      grad != nullptr,
      common::errors::Fatal("Detected NULL grad"
                            "Please check if you have manually cleared"
                            "the grad inside autograd_meta"));
  *grad = src;
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

int tensor_properties_set_stop_gradient(TensorObject* self,
                                        PyObject* value,
                                        void* closure) {
  EAGER_TRY
  auto meta = egr::EagerUtils::autograd_meta(&self->tensor);
  meta->SetStopGradient(CastPyArg2AttrBoolean(value, 0));
  if (!meta->GradNode()) {
    meta->SetGradNode(std::make_shared<egr::GradNodeAccumulation>(meta));
  }
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

PyDoc_STRVAR(tensor_persistable__doc__,  // NOLINT
             R"DOC(persistable

Tensor's persistable.

Returns:
    bool: persistable.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor(1.0, stop_gradient=False)
        >>> print(x.persistable)
        False

        >>> x. persistable = True
        >>> print(x.persistable)
        True
)DOC");

PyObject* tensor_properties_get_persistable(TensorObject* self, void* closure) {
  EAGER_TRY
  auto meta = egr::EagerUtils::autograd_meta(&self->tensor);
  return ToPyObject(meta->Persistable());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_persistable(TensorObject* self,
                                      PyObject* value,
                                      void* closure) {
  EAGER_TRY
  auto meta = egr::EagerUtils::autograd_meta(&self->tensor);
  meta->SetPersistable(CastPyArg2AttrBoolean(value, 0));
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

PyDoc_STRVAR(tensor_process_mesh__doc__,  // NOLINT
             R"DOC(process_mesh

Get process_mesh property from shard tensor.

Returns:
    core.ProcessMesh: the process mesh of shard tensor

Examples:
    .. code-block:: python

        >>> # doctest: +REQUIRES(env:DISTRIBUTED)
        >>> import paddle
        >>> import paddle.distributed as dist
        >>> from paddle.base import core

        >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"])

        >>> a = paddle.to_tensor([[1,2,3],
        ...                       [5,6,7]])
        >>> d_tensor = dist.shard_tensor(a, mesh, [core.Shard(0), core.Shard(1)])

        >>> print(d_tensor.process_mesh)

)DOC");

PyObject* tensor_properties_get_process_mesh(TensorObject* self,
                                             void* closure) {
  EAGER_TRY
  if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
    return ToPyObject(&dist_tensor->process_mesh());
#else
    PADDLE_THROW(common::errors::Unavailable(
        "The `process_mesh` property of (Dist)Tensor is not supported in the "
        "current PaddlePaddle, please recompile and install PaddlePaddle with "
        "the "
        "option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_placements__doc__,  // NOLINT
             R"DOC(placements

Get placements property from shard tensor.

Returns:
    List[core.Placement]: the process mesh of shard tensor

Examples:
    .. code-block:: python

        >>> # doctest: +REQUIRES(env:DISTRIBUTED)
        >>> import paddle
        >>> import paddle.distributed as dist
        >>> from paddle.base import core

        >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"])

        >>> a = paddle.to_tensor([[1,2,3],
        ...                       [5,6,7]])
        >>> d_tensor = dist.shard_tensor(a, mesh, [core.Shard(0), core.Shard(1)])

        >>> print(d_tensor.placements)

)DOC");

PyObject* tensor_properties_get_placements(TensorObject* self, void* closure) {
  EAGER_TRY
  if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
    return ToPyObject(dist_tensor->placements());
#else
    PADDLE_THROW(common::errors::Unavailable(
        "The `placements()` property of (Dist)Tensor is not supported in the "
        "current PaddlePaddle, please recompile and installPaddlePaddle with "
        "the "
        "option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_num_shard__doc__,  // NOLINT
             R"DOC(num_shard

Tensor's num_shard.

Returns:
    int64_t: Tensor's num_shard.

Examples:
    .. code-block:: python

        >>> # doctest: +REQUIRES(env:DISTRIBUTED)
        >>> import paddle
        >>> import paddle.distributed as dist
        >>> from paddle.base import core

        >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"])

        >>> a = paddle.to_tensor([[1,2,3],
        ...                       [5,6,7]])
        >>> d_tensor = paddle.Tensor(a, [core.Shard(0), core.Shard(1)])

        >>> print(d_tensor.num_shard) # 4

)DOC");

PyObject* tensor_properties_get_num_shard(TensorObject* self, void* closure) {
  EAGER_TRY
  if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
    return ToPyObject(dist_tensor->num_shard());
#else
    PADDLE_THROW(common::errors::Unavailable(
        "The `num_shard` property of (Dist)Tensor is not supported in the "
        "current PaddlePaddle, please recompile and install PaddlePaddle with "
        "the "
        "option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_local_shape(TensorObject* self, void* closure) {
  EAGER_TRY
  if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
    return ToPyObject(common::vectorize<int64_t>(dist_tensor->local_dims()));
#else
    PADDLE_THROW(common::errors::Unavailable(
        "The `_local_shape` property of (Dist)Tensor is not supported "
        "in the current PaddlePaddle, please recompile and install "
        "PaddlePaddle "
        "with the option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_shape__doc__,  // NOLINT
             R"DOC(shape

Tensor's shape.

Returns:
    List: shape.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor(1.0, stop_gradient=False)
        >>> print(x.shape)
        []
)DOC");

PyObject* tensor_properties_get_shape(TensorObject* self, void* closure) {
  EAGER_TRY
  std::vector<int64_t> value;
  if (!self->tensor.defined()) {
    return ToPyObject(value);
  }
  if (egr::IsVariableCompatTensor(self->tensor)) {
    auto* var_tensor = static_cast<const egr::VariableCompatTensor*>(
        self->tensor.impl().get());
    if (var_tensor->IsType<phi::Vocab>()) {
      value.emplace_back(
          static_cast<int64_t>(var_tensor->Get<phi::Vocab>().size()));
    } else if (var_tensor->IsType<phi::Strings>()) {
      value.emplace_back(
          static_cast<int64_t>(var_tensor->Get<phi::Strings>().size()));
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "VariableCompatTensor only support get shape from Vocab or "
          "Strings."));
    }
  } else {
    auto ddim = self->tensor.shape();
    size_t rank = static_cast<size_t>(ddim.size());
    value.resize(rank);
    for (size_t i = 0; i < rank; i++) {
      value[i] = ddim[i];
    }
  }
  if (!egr::IsVariableCompatTensor(self->tensor)) {
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    auto default_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDefaultLayout();
    bool change_dim =
        (desired_layout != default_layout &&
         self->tensor.layout() == desired_layout && value.size() == 4);
    VLOG(6) << "eager_properties 'Shape' method, layout autotune "
            << " desired_layout: " << desired_layout
            << " default_layout: " << default_layout
            << " tensor layout: " << self->tensor.layout()
            << " tensor's shape size is : " << value.size();
    std::vector<int64_t> dims = value;
    if (change_dim && common::DataLayoutToString(desired_layout) == "NCHW") {
      // NCHW -> NHWC
      VLOG(6) << "layout autotune get Shape from NCHW -> NHWC " << value[0]
              << " " << value[1] << " " << value[2] << " " << value[3] << " to "
              << dims[0] << " " << dims[2] << " " << dims[3] << " " << dims[1];
      value[0] = dims[0];
      value[1] = dims[2];
      value[2] = dims[3];
      value[3] = dims[1];
    } else if (change_dim &&
               common::DataLayoutToString(desired_layout) == "NHWC") {
      // NHWC -> NCHW
      VLOG(6) << "layout autotune get Shape from NHWC -> NCHW " << value[0]
              << " " << value[1] << " " << value[2] << " " << value[3] << " to "
              << dims[0] << " " << dims[3] << " " << dims[1] << " " << dims[2]
              << " " << dims[1];
      value[0] = dims[0];
      value[1] = dims[3];
      value[2] = dims[1];
      value[3] = dims[2];
    }
  }

  return ToPyObject(value);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_strides__doc__,  // NOLINT
             R"DOC(strides

Tensor's strides.

Returns:
    List: strides.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1, 2, 3])
        >>> y = x[1]
        >>> print(y.strides)
        []
)DOC");

PyObject* tensor_properties_get_strides(TensorObject* self, void* closure) {
  EAGER_TRY
  std::vector<int64_t> value;
  if (!self->tensor.defined() ||
      (!self->tensor.is_dense_tensor() && !self->tensor.is_dist_tensor())) {
    return ToPyObject(value);
  }

  auto stride = self->tensor.strides();
  size_t rank = static_cast<size_t>(stride.size());
  value.resize(rank);

  for (int i = 0; i < static_cast<int>(rank); i++) {
    value[i] = stride[i];
  }

  return ToPyObject(value);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_offset__doc__,  // NOLINT
             R"DOC(offset

The address of the first element relative to the offset of the video memory.

Returns:
    int: offset.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1, 2, 3])
        >>> y = x[1]
        >>> print(y.offset)
        8
)DOC");
PyObject* tensor_properties_get_offset(TensorObject* self, void* closure) {
  EAGER_TRY
  if (!self->tensor.defined() ||
      (!self->tensor.is_dense_tensor() && !self->tensor.is_dist_tensor())) {
    RETURN_PY_NONE;
  }

  size_t offset = 0;
  if (self->tensor.is_dense_tensor()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
    if (dense_tensor == nullptr) {
      RETURN_PY_NONE;
    }
    offset = dense_tensor->offset();
  } else if (self->tensor.is_dist_tensor()) {
    auto dist_tensor = std::dynamic_pointer_cast<phi::distributed::DistTensor>(
        self->tensor.impl());
    if (dist_tensor == nullptr) {
      RETURN_PY_NONE;
    }
    offset = dist_tensor->value().offset();
  }

  return ToPyObject(offset);

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_layout__doc__,  // NOLINT
             R"DOC(layout

Tensor's memory layout.

Returns:
    Layout: layout.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1, 2, 3])
        >>> print(x.layout)
        NCHW
)DOC");
PyObject* tensor_properties_get_layout(TensorObject* self, void* closure) {
  EAGER_TRY
  std::string layout = "";
  if (!self->tensor.defined()) {
    return ToPyObject(layout);
  }

  if (egr::IsVariableCompatTensor(self->tensor)) {
    VLOG(3) << "VariableCompatTensor does not support `layout` method.";
    return ToPyObject(layout);
  } else {
    return ToPyObject(common::DataLayoutToString(self->tensor.layout()));
  }

  return ToPyObject(layout);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_place__doc__,  // NOLINT
             R"DOC(place

The device Tensor's memory locate.

Returns:
    Place: place.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1, 2, 3])
        >>> print(x.place)
        Place(cpu)
)DOC");
PyObject* tensor_properties_get_place(TensorObject* self, void* closure) {
  EAGER_TRY
  return ToPyObject(self->tensor.place());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_place_str(TensorObject* self, void* closure) {
  EAGER_TRY
  std::stringstream ostr;
  ostr << self->tensor.place();
  return ToPyObject(ostr.str());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_placements_str(TensorObject* self,
                                               void* closure) {
  EAGER_TRY
  if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());

    std::stringstream ostr;
    ostr << "[";

    bool isFirst = true;
    for (const auto& p : dist_tensor->placements()) {
      if (p) {
        if (!isFirst) {
          ostr << ", ";
        }
        ostr << p->to_string();
        isFirst = false;
      }
    }
    ostr << "]";
    return ToPyObject(ostr.str());
#else
    PADDLE_THROW(common::errors::Unavailable(
        "The `placements()` property of (Dist)Tensor is not supported in the "
        "current PaddlePaddle, please recompile and installPaddlePaddle with "
        "the "
        "option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_dtype__doc__,  // NOLINT
             R"DOC(dtype

Tensor's data type.

Returns:
    paddle dtype: dtype.

Examples:
    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([1, 2, 3])
        >>> print(x.dtype)
        paddle.int64
)DOC");
PyObject* tensor_properties_get_dtype(TensorObject* self, void* closure) {
  EAGER_TRY
  if (FLAGS_enable_pir_api) {
    if (!self->tensor.defined()) {
      // be same to old dygraph
      return ToPyObject(phi::DataType::FLOAT32);
    }
    if (egr::IsVariableCompatTensor(self->tensor)) {
      auto* var_tensor = static_cast<const egr::VariableCompatTensor*>(
          self->tensor.impl().get());
      if (var_tensor->IsType<phi::Vocab>()) {
        return ToPyObject(phi::DataType::UNDEFINED);
      } else if (var_tensor->IsType<phi::Strings>()) {
        return ToPyObject(phi::DataType::PSTRING);
      } else {
        PADDLE_THROW(common::errors::Unavailable(
            "VariableCompatTensor only support get shape from Vocab or "
            "Strings."));
      }
    } else {
      return ToPyObject(self->tensor.type());
    }
  } else {
    if (!self->tensor.defined()) {
      // be same to old dygraph
      return ToPyObject(framework::proto::VarType::FP32);
    }
    if (egr::IsVariableCompatTensor(self->tensor)) {
      auto* var_tensor = static_cast<const egr::VariableCompatTensor*>(
          self->tensor.impl().get());
      if (var_tensor->IsType<phi::Vocab>()) {
        return ToPyObject(framework::proto::VarType::RAW);
      } else if (var_tensor->IsType<phi::Strings>()) {
        return ToPyObject(framework::proto::VarType::STRING);
      } else {
        PADDLE_THROW(common::errors::Unavailable(
            "VariableCompatTensor only support get shape from Vocab or "
            "Strings."));
      }
    } else {
      return ToPyObject(
          paddle::framework::TransToProtoVarType(self->tensor.type()));
    }
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_grad_fn(TensorObject* self, void* closure) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    // Handle undefined tensors if necessary; otherwise, return nullptr or an
    // appropriate PyObject. In this case, I will return Py_None.
    Py_INCREF(Py_None);
    return Py_None;
  }

  // Get GradNode from the tensor
  auto meta = egr::EagerUtils::nullable_autograd_meta(
      self->tensor);  // If meta exists, get the GradNode

  if (meta) {
    // Get the GradNode from meta
    auto grad_node_ptr = meta->GetMutableGradNode();
    if (!grad_node_ptr) {
      Py_INCREF(Py_None);
      return Py_None;
    }

    PyObject* py_grad_node = ToPyObject(grad_node_ptr);

    return py_grad_node;

  } else {
    // If meta does not exist, return an appropriate Python object (e.g., None
    // or a special value).
    Py_INCREF(Py_None);
    return Py_None;
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties___dict__(TensorObject* self, void*) {
  EAGER_TRY
  if (self->dict == nullptr) {
    self->dict = PyDict_New();
  }
  Py_INCREF(self->dict);
  return self->dict;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

struct PyGetSetDef variable_properties[] = {  // NOLINT
    {"data",
     (getter)tensor_properties_get_data,
     (setter)tensor_properties_set_data,
     tensor_data__doc__,
     nullptr},
    {"grad",
     (getter)tensor_properties_get_grad,
     (setter)tensor_properties_set_grad,
     tensor_grad__doc__,
     nullptr},
    {"grad_",
     (getter)tensor_properties_get_grad,
     (setter)tensor_properties_set_grad_,
     nullptr,
     nullptr},
    {"name",
     (getter)tensor_properties_get_name,
     (setter)tensor_properties_set_name,
     tensor_name__doc__,
     nullptr},
    {"stop_gradient",
     (getter)tensor_properties_get_stop_gradient,
     (setter)tensor_properties_set_stop_gradient,
     tensor_stop_gradient__doc__,
     nullptr},
    {"persistable",
     (getter)tensor_properties_get_persistable,
     (setter)tensor_properties_set_persistable,
     tensor_persistable__doc__,
     nullptr},
    {"_local_shape",
     (getter)tensor_properties_get_local_shape,
     nullptr,
     nullptr,
     nullptr},
    {"shape",
     (getter)tensor_properties_get_shape,
     nullptr,
     tensor_shape__doc__,
     nullptr},
    {"layout",
     (getter)tensor_properties_get_layout,
     nullptr,
     tensor_layout__doc__,
     nullptr},
    {"strides",
     (getter)tensor_properties_get_strides,
     nullptr,
     tensor_strides__doc__,
     nullptr},
    {"place",
     (getter)tensor_properties_get_place,
     nullptr,
     tensor_place__doc__,
     nullptr},
    {"offset",
     (getter)tensor_properties_get_offset,
     nullptr,
     tensor_offset__doc__,
     nullptr},
    {"process_mesh",
     (getter)tensor_properties_get_process_mesh,
     nullptr,
     tensor_process_mesh__doc__,
     nullptr},
    {"placements",
     (getter)tensor_properties_get_placements,
     nullptr,
     tensor_placements__doc__,
     nullptr},
    {"num_shard",
     (getter)tensor_properties_get_num_shard,
     nullptr,
     tensor_num_shard__doc__,
     nullptr},
    {"_place_str",
     (getter)tensor_properties_get_place_str,
     nullptr,
     nullptr,
     nullptr},
    {"_placements_str",
     (getter)tensor_properties_get_placements_str,
     nullptr,
     nullptr,
     nullptr},
    {"dtype",
     (getter)tensor_properties_get_dtype,
     nullptr,
     tensor_dtype__doc__,
     nullptr},
    {"type",
     (getter)tensor_properties_get_type,
     nullptr,
     tensor_type__doc__,
     nullptr},
    {"is_leaf",
     (getter)tensor_properties_is_leaf,
     nullptr,
     tensor_is_leaf__doc__,
     nullptr},
    {"grad_fn",
     (getter)tensor_properties_get_grad_fn,
     nullptr,
     nullptr,
     nullptr},
    {"__dict__", (getter)tensor_properties___dict__, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

// variable_properties for core.eager.StringTensor
struct PyGetSetDef string_tensor_variable_properties[] = {  // NOLINT
    {"name",
     (getter)tensor_properties_get_name,
     (setter)tensor_properties_set_name,
     nullptr,
     nullptr},
    {"shape", (getter)tensor_properties_get_shape, nullptr, nullptr, nullptr},
    {"layout", (getter)tensor_properties_get_layout, nullptr, nullptr, nullptr},
    {"place", (getter)tensor_properties_get_place, nullptr, nullptr, nullptr},
    {"_place_str",
     (getter)tensor_properties_get_place_str,
     nullptr,
     nullptr,
     nullptr},
    {"__dict__", (getter)tensor_properties___dict__, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

}  // namespace paddle::pybind
