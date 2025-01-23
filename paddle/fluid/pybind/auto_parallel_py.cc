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

#include <Python.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <utility>

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dist_tensor_spec.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/pybind/auto_parallel_py.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/reduce_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/auto_parallel/device_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_mapper.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/placement_types.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/nd_mesh_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/x_to_r_reshard_function.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/optional.h"
#include "paddle/utils/pybind.h"

#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#endif

namespace py = pybind11;  // NOLINT

namespace paddle::pybind {

static bool PyCheckInteger(PyObject *obj) {
#if PY_VERSION_HEX < 0x03000000
  return (PyLong_Check(obj) || PyInt_Check(obj)) && !PyBool_Check(obj);
#else
  return PyLong_Check(obj) && !PyBool_Check(obj);
#endif
}

using paddle::distributed::auto_parallel::DistTensorSpec;
using paddle::distributed::auto_parallel::kDefault;
using paddle::distributed::auto_parallel::OperatorDistAttr;
using paddle::framework::BlockDesc;
using paddle::framework::OpDesc;
using paddle::framework::VarDesc;
using phi::distributed::ArgDistAttr;
using phi::distributed::ProcessMesh;
using phi::distributed::TensorDistAttr;
using phi::distributed::auto_parallel::Device;
using phi::distributed::auto_parallel::DeviceCapability;
using phi::distributed::auto_parallel::DeviceMesh;
using phi::distributed::auto_parallel::DistributedMapper;
using phi::distributed::auto_parallel::Link;
using phi::distributed::auto_parallel::LinkCapability;
using phi::distributed::auto_parallel::Machine;

PyTypeObject *g_tensor_dist_attr_pytype = nullptr;
PyTypeObject *g_dist_tensor_spec_pytype = nullptr;
PyTypeObject *g_process_mesh_pytype = nullptr;
PyTypeObject *g_placement_shard_pytype = nullptr;
PyTypeObject *g_placement_replicated_pytype = nullptr;
PyTypeObject *g_placement_partial_pytype = nullptr;

constexpr const char *infer_spmd_string = "infer_spmd";

static inline const ProcessMesh *get_tensor_process_mesh(
    const TensorDistAttr &self) {
  if (self.process_mesh().empty()) {
    return nullptr;
  } else {
    return &self.process_mesh();
  }
}

static inline void set_tensor_process_mesh(TensorDistAttr *self,
                                           const ProcessMesh *process_mesh) {
  if (process_mesh) {
    self->set_process_mesh(*process_mesh);
  } else {
    self->set_process_mesh(ProcessMesh());
  }
}

static inline const ProcessMesh *get_operator_process_mesh(
    const OperatorDistAttr &self) {
  if (self.process_mesh().empty()) {
    return nullptr;
  } else {
    return &self.process_mesh();
  }
}

static inline void set_operator_process_mesh(OperatorDistAttr *self,
                                             const ProcessMesh *process_mesh) {
  if (process_mesh) {
    self->set_process_mesh(*process_mesh);
  } else {
    self->set_process_mesh(ProcessMesh());
  }
}

static inline void reset_tensor_dist_attr(TensorDistAttr *dist_attr) {
  dist_attr->set_process_mesh(ProcessMesh());
  std::vector<int64_t> dims_mapping(dist_attr->dims_mapping().size(), -1);
  dist_attr->set_dims_mapping(dims_mapping);
  dist_attr->clear_annotated();
}

static inline void reset_operator_dist_attr(OperatorDistAttr *dist_attr) {
  for (auto &item : dist_attr->input_dist_attrs()) {
    reset_tensor_dist_attr(&item.second);
  }
  for (auto &item : dist_attr->output_dist_attrs()) {
    reset_tensor_dist_attr(&item.second);
  }
  dist_attr->set_impl_type(kDefault);
  dist_attr->set_impl_idx(0);
  dist_attr->set_chunk_id(0);
  dist_attr->clear_annotated();
}

static std::pair<std::vector<ArgDistAttr>, std::vector<ArgDistAttr>>
infer_forward(const phi::distributed::SpmdRule &self, const py::args &args);
static std::pair<std::vector<ArgDistAttr>, std::vector<ArgDistAttr>>
infer_backward(const phi::distributed::SpmdRule &self, const py::args &args);

void BindAutoParallel(py::module *m) {
  auto ReshardFunction =
      py::class_<phi::distributed::ReshardFunction>(*m, "ReshardFunction")
          .def(
              "is_suitable",
              [](phi::distributed::ReshardFunction &self,
                 py::handle py_tensor,
                 const phi::distributed::TensorDistAttr &dist_attr) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dist =
                    std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                        tensor.impl());
                return self.IsSuitable(*p_dist, dist_attr);
              },
              py::call_guard<py::gil_scoped_release>())
          .def(
              "eval",
              [](phi::distributed::ReshardFunction &self,
                 phi::DeviceContext *dev_ctx,
                 py::handle py_tensor,
                 const phi::distributed::TensorDistAttr &dist_attr) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dist =
                    std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                        tensor.impl());
                auto res_dist = self.Eval(dev_ctx, *p_dist, dist_attr);
                return paddle::Tensor(res_dist);
              },
              py::call_guard<py::gil_scoped_release>());

  py::class_<phi::distributed::RToSReshardFunction>(
      *m, "RToSReshardFunction", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::RToSReshardFunctionCrossMesh>(
      *m, "RToSReshardFunctionCrossMesh", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::SToRReshardFunction>(
      *m, "SToRReshardFunction", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::SToRReshardFunctionCrossMesh>(
      *m, "SToRReshardFunctionCrossMesh", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::RToPReshardFunction>(
      *m, "RToPReshardFunction", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::RToPReshardFunctionCrossMesh>(
      *m, "RToPReshardFunctionCrossMesh", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::PToRReshardFunction>(
      *m, "PToRReshardFunction", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::PToRReshardFunctionCrossMesh>(
      *m, "PToRReshardFunctionCrossMesh", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::SToSReshardFunction>(
      *m, "SToSReshardFunction", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::SToPReshardFunction>(
      *m, "SToPReshardFunction", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::PToSReshardFunction>(
      *m, "PToSReshardFunction", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::XToRShrinkReshardFunction>(
      *m, "XToRShrinkReshardFunction", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::SameNdMeshReshardFunction>(
      *m, "SameNdMeshReshardFunction", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::SameStatusReshardFunction>(
      *m, "SameStatusReshardFunction", ReshardFunction)
      .def(py::init<>());

  auto process_mesh =
      py::class_<ProcessMesh>(*m, "ProcessMesh")
          .def(py::init<>())
          .def(py::init<const std::vector<int64_t> &,
                        const std::vector<int64_t> &,
                        const std::vector<std::string> &>(),
               py::arg("shape"),
               py::arg("process_ids"),
               py::arg("dim_names"))
          .def_property_readonly("shape", &ProcessMesh::shape)
          .def_property_readonly("process_ids", &ProcessMesh::process_ids)
          .def_property_readonly("dim_names", &ProcessMesh::dim_names)
          .def_property_readonly("size", &ProcessMesh::size)
          .def_property_readonly("ndim", &ProcessMesh::ndim)
          .def("dim_size",
               static_cast<int64_t (ProcessMesh::*)(int64_t) const>(
                   &ProcessMesh::dim_size))
          .def("dim_size",
               static_cast<int64_t (ProcessMesh::*)(const std::string &) const>(
                   &ProcessMesh::dim_size))
          .def("empty", &ProcessMesh::empty)
          .def("contains", &ProcessMesh::contains)
          .def(py::self == py::self)  // NOLINT
          .def(py::self != py::self)  // NOLINT
          .def("__copy__",
               [](const ProcessMesh &self) { return ProcessMesh(self); })
          .def(
              "__deepcopy__",
              [](const ProcessMesh &self, py::dict) {
                return ProcessMesh(self);
              },
              py::arg("memo"))
          .def("__hash__", &ProcessMesh::hash)
          .def("__str__", &ProcessMesh::to_string);

  g_process_mesh_pytype = reinterpret_cast<PyTypeObject *>(process_mesh.ptr());

  py::class_<DeviceCapability>(*m, "DeviceCapability")
      .def(py::init<>())
      .def_readwrite("sflops", &DeviceCapability::single_precision_flops)
      .def_readwrite("dflops", &DeviceCapability::double_precision_flops)
      .def_readwrite("memory", &DeviceCapability::memory_size_in_bytes)
      .def_readwrite("rate", &DeviceCapability::clock_rate_in_ghz)
      .def("__str__", &DeviceCapability::to_string);

  py::class_<Device>(*m, "Device")
      .def(py::init<int64_t, int64_t, int64_t, const std::string &>(),
           py::arg("global_id"),
           py::arg("local_id"),
           py::arg("machine_id"),
           py::arg("type"))
      .def_property_readonly("global_id", &Device::global_id)
      .def_property_readonly("local_id", &Device::local_id)
      .def_property_readonly("machine_id", &Device::machine_id)
      .def_property_readonly("type", &Device::type)
      .def_property("capability", &Device::capability, &Device::set_capability)
      .def(py::self == py::self)  // NOLINT
      .def(py::self != py::self)  // NOLINT
      .def("__str__", &Device::to_string);

  py::class_<LinkCapability>(*m, "LinkCapability")
      .def(py::init<>())
      .def_readwrite("bandwidth", &LinkCapability::bandwidth)
      .def_readwrite("latency", &LinkCapability::latency)
      .def("__str__", &LinkCapability::to_string);

  py::class_<Link>(*m, "Link")
      .def(py::init<int64_t, int64_t, const std::string &>(),
           py::arg("source_id"),
           py::arg("target_id"),
           py::arg("type"))
      .def_property_readonly("source_id", &Link::source_id)
      .def_property_readonly("target_id", &Link::target_id)
      .def_property_readonly("type", &Link::type)
      .def_property("capability", &Link::capability, &Link::set_capability)
      .def(py::self == py::self)  // NOLINT
      .def(py::self != py::self)  // NOLINT
      .def("__str__", &Link::to_string);

  py::class_<Machine>(*m, "Machine")
      .def_property_readonly("id", &Machine::id)
      .def_property_readonly("devices", &Machine::devices)
      .def_property_readonly("links", &Machine::links)
      .def("device", &Machine::device)
      .def("link", &Machine::link)
      .def("contains", &Machine::contains)
      .def("__str__", &Machine::to_string);

  py::class_<DeviceMesh>(*m, "DeviceMesh")
      .def(py::init<const std::string &,
                    const std::vector<int64_t> &,
                    const std::vector<int64_t> &,
                    const std::vector<std::string> &>(),
           py::arg("name"),
           py::arg("shape"),
           py::arg("device_ids"),
           py::arg("dim_names"))
      .def_property_readonly("name", &DeviceMesh::name)
      .def_property_readonly("shape", &DeviceMesh::shape)
      .def_property_readonly("device_ids", &DeviceMesh::device_ids)
      .def_property_readonly("dim_names", &DeviceMesh::dim_names)
      .def_property_readonly("device_type", &DeviceMesh::device_type)
      .def_property_readonly("size", &DeviceMesh::size)
      .def_property_readonly("ndim", &DeviceMesh::ndim)
      .def_property_readonly("devices", &DeviceMesh::devices)
      .def_property_readonly("links", &DeviceMesh::links)
      .def_property_readonly("machines", &DeviceMesh::machines)
      .def("device", &DeviceMesh::device)
      .def("link", &DeviceMesh::link)
      .def("machine", &DeviceMesh::machine)
      .def("empty", &DeviceMesh::empty)
      .def("contains", &DeviceMesh::contains)
      .def("add_device", &DeviceMesh::add_device)
      .def("add_link", &DeviceMesh::add_link)
      .def("dim_size",
           static_cast<int64_t (DeviceMesh::*)(int64_t) const>(
               &DeviceMesh::dim_size))
      .def("dim_size",
           static_cast<int64_t (DeviceMesh::*)(const std::string &) const>(
               &DeviceMesh::dim_size))
      .def(py::self == py::self)  // NOLINT
      .def(py::self != py::self)  // NOLINT
      .def("__copy__",
           [](const TensorDistAttr &self) { return TensorDistAttr(self); })
      .def(
          "__deepcopy__",
          [](const TensorDistAttr &self, py::dict) {
            return TensorDistAttr(self);
          },
          py::arg("memo"))
      .def("__str__", &DeviceMesh::to_string);

  py::enum_<phi::ReduceType>(*m, "ReduceType", R"DOC(
    Specify the type of operation used for paddle.distributed.Partial().
    It should be one of the following values:

        - ReduceType.kRedSum
        - ReduceType.kRedMax
        - ReduceType.kRedMin
        - ReduceType.kRedProd
        - ReduceType.kRedAvg
        - ReduceType.kRedAny
        - ReduceType.kRedAll

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist
            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            >>> a = paddle.ones([10, 20])
            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> # distributed tensor
            >>> d_tensor = dist.shard_tensor(a, mesh, [dist.Partial(dist.ReduceType.kRedSum)])

      )DOC")
      .value("kRedSum", phi::ReduceType::kRedSum)
      .value("kRedMax", phi::ReduceType::kRedMax)
      .value("kRedMin", phi::ReduceType::kRedMin)
      .value("kRedProd", phi::ReduceType::kRedProd)
      .value("kRedAvg", phi::ReduceType::kRedAvg)
      .value("kRedAny", phi::ReduceType::kRedAny)
      .value("kRedAll", phi::ReduceType::kRedAll);

  auto Placement =
      py::class_<phi::distributed::Placement,
                 std::shared_ptr<phi::distributed::Placement>>(
          *m, "Placement", R"DOC(
        The `Placement` is base class that describes how to place the tensor on ProcessMesh. it has three subclass: `Replicate`, `Shard` and `Partial`.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed as dist
                >>> placements = [dist.Replicate(), dist.Shard(0), dist.Partial()]
                >>> for p in placements:
                >>>     if isinstance(p, dist.Placement):
                >>>         if p.is_replicated():
                >>>             print("replicate.")
                >>>         elif p.is_shard():
                >>>             print("shard.")
                >>>         elif p.is_partial():
                >>>             print("partial.")

      )DOC")
          .def(py::init<>())
          .def("is_shard",
               &phi::distributed::Placement::is_shard,
               py::arg("dim") = std::nullopt)
          .def("is_replicated", &phi::distributed::Placement::is_replicated)
          .def("is_partial", &phi::distributed::Placement::is_partial)
          .def("__hash__", &phi::distributed::Placement::hash)
          .def("__str__", &phi::distributed::Placement::to_string)
          .def("__repr__", &phi::distributed::Placement::to_string)
          .def("__copy__",
               [](const phi::distributed::Placement &self) {
                 return phi::distributed::Placement(self);
               })
          .def(
              "__deepcopy__",
              [](const phi::distributed::Placement &self, py::dict) {
                return phi::distributed::Placement(self);
              },
              py::arg("memo"))
          .def(py::self == py::self)   // NOLINT
          .def(py::self != py::self);  // NOLINT

  auto Shard = py::class_<phi::distributed::Shard,
                          std::shared_ptr<phi::distributed::Shard>>(
                   *m, "Shard", Placement, R"DOC(
               The `Shard` describes how `Tensor` splitted across multiple devices according to specified dimensions.

               Parameters:
                   dim (int): specify the slicing dimension of the tensor.

               Examples:
                   .. code-block:: python

                       >>> import paddle
                       >>> import paddle.distributed as dist
                       >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=['x', 'y'])
                       >>> a = paddle.to_tensor([[1,2,3],[5,6,7]])
                       >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                       >>> # distributed tensor
                       >>> d_tensor = dist.shard_tensor(a, mesh, [dist.Shard(0), dist.Shard(1)])

               )DOC")
                   .def(py::init([](int64_t dim) {
                     return std::make_shared<phi::distributed::Shard>(dim);
                   }))
                   .def("get_dim", &phi::distributed::Shard::get_dim)
                   .def("__hash__", &phi::distributed::Shard::hash)
                   .def("__str__", &phi::distributed::Shard::to_string)
                   .def("__repr__", &phi::distributed::Shard::to_string)
                   .def("__copy__",
                        [](const phi::distributed::Shard &self) {
                          return phi::distributed::Shard(self);
                        })
                   .def(
                       "__deepcopy__",
                       [](const phi::distributed::Shard &self, py::dict) {
                         return phi::distributed::Shard(self);
                       },
                       py::arg("memo"))
                   .def(py::self == py::self)   // NOLINT
                   .def(py::self != py::self);  // NOLINT

  auto Replicate =
      py::class_<phi::distributed::Replicate,
                 std::shared_ptr<phi::distributed::Replicate>>(
          *m, "Replicate", Placement, R"DOC(
                   The `Replicate` describes the tensor placed repeatedly on ProcessMesh.

                   Examples:
                       .. code-block:: python

                           >>> import paddle
                           >>> import paddle.distributed as dist
                           >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
                           >>> a = paddle.ones([10, 20])
                           >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                           >>> # distributed tensor
                           >>> d_tensor = dist.shard_tensor(a, mesh, [dist.Replicate()])

                   )DOC")
          .def(py::init<>())
          .def("__hash__", &phi::distributed::Replicate::hash)
          .def("__str__", &phi::distributed::Replicate::to_string)
          .def("__repr__", &phi::distributed::Replicate::to_string)
          .def("__copy__",
               [](const phi::distributed::Replicate &self) {
                 return phi::distributed::Replicate(self);
               })
          .def(
              "__deepcopy__",
              [](const phi::distributed::Replicate &self, py::dict) {
                return phi::distributed::Replicate(self);
              },
              py::arg("memo"))
          .def(py::self == py::self)   // NOLINT
          .def(py::self != py::self);  // NOLINT

  auto Partial =
      py::class_<phi::distributed::Partial,
                 std::shared_ptr<phi::distributed::Partial>>(
          *m, "Partial", Placement, R"DOC(
                 The `Partial` describes `Tensor` across multiple devices, this type of tensor has the same shape but only a fraction of the value, which can be further reduce (e.g. sum/min/max) to obtain dist_tensor, often used as an intermediate representation.

                 Parameters:
                   reduce_type (paddle.distributed.ReduceType): the reduce type of the Partial state, default `paddle.distributed.ReduceType.kRedSum`.

                 Examples:
                     .. code-block:: python

                         >>> import paddle
                         >>> import paddle.distributed as dist
                         >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
                         >>> a = paddle.ones([10, 20])
                         >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                         >>> # distributed tensor
                         >>> d_tensor = dist.shard_tensor(a, mesh, [dist.Partial()])

                 )DOC")
          .def(py::init<phi::ReduceType>(),
               py::arg("reduce_type") = phi::ReduceType::kRedSum)
          .def("reduce_type", &phi::distributed::Partial::get_reduce_type)
          .def("__hash__", &phi::distributed::Partial::hash)
          .def("__str__", &phi::distributed::Partial::to_string)
          .def("__repr__", &phi::distributed::Partial::to_string)
          .def("__copy__",
               [](const phi::distributed::Partial &self) {
                 return phi::distributed::Partial(self);
               })
          .def(
              "__deepcopy__",
              [](const phi::distributed::Partial &self, py::dict) {
                return phi::distributed::Partial(self);
              },
              py::arg("memo"))
          .def(py::self == py::self)   // NOLINT
          .def(py::self != py::self);  // NOLINT

  g_placement_shard_pytype = reinterpret_cast<PyTypeObject *>(Shard.ptr());
  g_placement_replicated_pytype =
      reinterpret_cast<PyTypeObject *>(Replicate.ptr());
  g_placement_partial_pytype = reinterpret_cast<PyTypeObject *>(Partial.ptr());

  py::class_<TensorDistAttr> py_dist_attr(*m, "TensorDistAttr");
  g_tensor_dist_attr_pytype =
      reinterpret_cast<PyTypeObject *>(py_dist_attr.ptr());
  py_dist_attr.def(py::init<>())
      .def(py::init([](const VarDesc &var_desc) {
        auto shape =
            paddle::distributed::auto_parallel::get_tensor_shape(&var_desc);
        return std::make_unique<TensorDistAttr>(shape);
      }))
      .def(py::init<const TensorDistAttr &>())
      .def_property(
          "process_mesh", &get_tensor_process_mesh, &set_tensor_process_mesh)
      .def_property("dims_mapping",
                    &TensorDistAttr::dims_mapping,
                    &TensorDistAttr::set_dims_mapping)
      .def_property("batch_dim",
                    &TensorDistAttr::batch_dim,
                    &TensorDistAttr::set_batch_dim)
      .def_property(
          "chunk_id", &TensorDistAttr::chunk_id, &TensorDistAttr::set_chunk_id)
      .def_property("dynamic_dims",
                    &TensorDistAttr::dynamic_dims,
                    &TensorDistAttr::set_dynamic_dims)
      .def_property("annotated",
                    &TensorDistAttr::annotated,
                    &TensorDistAttr::set_annotated)
      .def("is_annotated", &TensorDistAttr::is_annotated)
      .def("mark_annotated", &TensorDistAttr::mark_annotated)
      .def("clear_annotated", &TensorDistAttr::clear_annotated)
      .def(
          "verify",
          [](TensorDistAttr &self, const VarDesc *tensor) {
            auto shape =
                paddle::distributed::auto_parallel::get_tensor_shape(tensor);
            return self.verify(shape);
          },
          py::arg("tensor") = static_cast<VarDesc *>(nullptr))
      .def("reset", &reset_tensor_dist_attr)
      .def("serialize_to_string",
           [](TensorDistAttr &self) {
             return py::bytes(self.serialize_to_string());
           })
      .def("parse_from_string", &TensorDistAttr::parse_from_string)
      .def(py::self == py::self)  // NOLINT
      .def(py::self != py::self)  // NOLINT
      .def("__copy__",
           [](const TensorDistAttr &self) { return TensorDistAttr(self); })
      .def(
          "__deepcopy__",
          [](const TensorDistAttr &self, py::dict) {
            return TensorDistAttr(self);
          },
          py::arg("memo"))
      .def("__str__", &TensorDistAttr::to_string)
      .def(
          "_is_partial", &TensorDistAttr::is_partial, py::arg("mesh_axis") = -1)
      .def("_partial_dims", &TensorDistAttr::partial_dims)
      .def("_clean_partial_dims", &TensorDistAttr::clean_partial_dims)
      .def("_set_partial_dims",
           [](TensorDistAttr &self, const std::vector<int64_t> &dims) {
             self.set_partial_status(dims);
           })
      .def("_clean_partial_status", &TensorDistAttr::clean_partial_status);

  py::class_<phi::distributed::SpmdRule>(*m, "SpmdRule")
      .def("infer_forward", &infer_forward)
      .def("infer_backward", &infer_backward);

  py::class_<DistTensorSpec> py_dist_tensor_spec(
      *m, "DistTensorSpec");  // TODO(ljz) remove and unify to DistTensor
  g_dist_tensor_spec_pytype =
      reinterpret_cast<PyTypeObject *>(py_dist_tensor_spec.ptr());
  py_dist_tensor_spec.def(py::init<>())
      .def(py::init<const DistTensorSpec &>())
      .def(py::init<const std::vector<int64_t> &, const TensorDistAttr &>())
      .def("dims_mapping", &DistTensorSpec::dims_mapping)
      .def("set_dims_mapping", &DistTensorSpec::set_dims_mapping)
      .def("process_mesh", &DistTensorSpec::process_mesh)
      .def("set_process_mesh", &DistTensorSpec::set_process_mesh)
      .def_property("shape", &DistTensorSpec::shape, &DistTensorSpec::set_shape)
      .def("__str__", &DistTensorSpec::to_string)
      .def("__copy__",
           [](const DistTensorSpec &self) { return DistTensorSpec(self); })
      .def(
          "__deepcopy__",
          [](const DistTensorSpec &self, py::dict) {
            return DistTensorSpec(self);
          },
          py::arg("memo"));

  py::class_<OperatorDistAttr>(*m, "OperatorDistAttr")
      .def(py::init<>())
      .def(py::init<const OpDesc &>())
      .def(py::init<const OperatorDistAttr &>())
      .def_property(
          "op_type", &OperatorDistAttr::op_type, &OperatorDistAttr::set_op_type)
      .def_property("process_mesh",
                    &get_operator_process_mesh,
                    &set_operator_process_mesh)
      .def_property("impl_type",
                    &OperatorDistAttr::impl_type,
                    &OperatorDistAttr::set_impl_type)
      .def_property("impl_idx",
                    &OperatorDistAttr::impl_idx,
                    &OperatorDistAttr::set_impl_idx)
      .def_property("chunk_id",
                    &OperatorDistAttr::chunk_id,
                    &OperatorDistAttr::set_chunk_id)
      .def_property("is_recompute",
                    &OperatorDistAttr::is_recompute,
                    &OperatorDistAttr::set_is_recompute)
      .def_property("execution_stream",
                    &OperatorDistAttr::execution_stream,
                    &OperatorDistAttr::set_execution_stream)
      .def_property("stream_priority",
                    &OperatorDistAttr::stream_priority,
                    &OperatorDistAttr::set_stream_priority)
      .def_property("scheduling_priority",
                    &OperatorDistAttr::scheduling_priority,
                    &OperatorDistAttr::set_scheduling_priority)
      .def_property("force_record_event",
                    &OperatorDistAttr::force_record_event,
                    &OperatorDistAttr::set_force_record_event)
      .def_property("events_to_wait",
                    &OperatorDistAttr::events_to_wait,
                    &OperatorDistAttr::set_events_to_wait,
                    pybind11::return_value_policy::reference)
      .def_property("event_to_record",
                    &OperatorDistAttr::event_to_record,
                    &OperatorDistAttr::set_event_to_record)
      .def_property("annotated",
                    &OperatorDistAttr::annotated,
                    &OperatorDistAttr::set_annotated)
      .def_property(
          "inputs_dist_attrs",
          static_cast<std::map<std::string, TensorDistAttr> &(
              OperatorDistAttr::*)()>(&OperatorDistAttr::input_dist_attrs),
          &OperatorDistAttr::set_input_dist_attrs)
      .def_property(
          "outputs_dist_attrs",
          static_cast<std::map<std::string, TensorDistAttr> &(
              OperatorDistAttr::*)()>(&OperatorDistAttr::output_dist_attrs),
          &OperatorDistAttr::set_output_dist_attrs)
      .def_property("run_time_us",
                    &OperatorDistAttr::run_time_us,
                    &OperatorDistAttr::set_run_time_us)
      .def("get_input_dist_attr",
           static_cast<TensorDistAttr &(
               OperatorDistAttr::*)(const std::string &)>(
               &OperatorDistAttr::input_dist_attr),
           py::return_value_policy::reference)
      .def("get_output_dist_attr",
           static_cast<TensorDistAttr &(
               OperatorDistAttr::*)(const std::string &)>(
               &OperatorDistAttr::output_dist_attr),
           py::return_value_policy::reference)
      .def("set_input_dist_attr", &OperatorDistAttr::set_input_dist_attr)
      .def("set_output_dist_attr", &OperatorDistAttr::set_output_dist_attr)
      .def("del_input_dist_attr",  // TODO(aoyulong): move into dist_attr.cc
           [](OperatorDistAttr &self, const std::string &name) {
             self.input_dist_attrs().erase(name);
           })
      .def("del_output_dist_attr",  // TODO(aoyulong): move into dist_attr.cc
           [](OperatorDistAttr &self, const std::string &name) {
             self.output_dist_attrs().erase(name);
           })
      .def("is_annotated", &OperatorDistAttr::is_annotated)
      .def("mark_annotated", &OperatorDistAttr::mark_annotated)
      .def("clear_annotated", &OperatorDistAttr::clear_annotated)
      .def("get_input_dims_mapping",
           &OperatorDistAttr::input_dims_mapping,
           py::return_value_policy::reference)
      .def("set_input_dims_mapping", &OperatorDistAttr::set_input_dims_mapping)
      .def("get_output_dims_mapping",
           &OperatorDistAttr::output_dims_mapping,
           py::return_value_policy::reference)
      .def("set_output_dims_mapping",
           &OperatorDistAttr::set_output_dims_mapping)
      .def("verify",
           &OperatorDistAttr::verify,
           py::arg("op") = static_cast<OpDesc *>(nullptr))
      .def("is_annotated_input_dims_mapping",
           [](const OperatorDistAttr &self, const std::string &name) {
             return self.input_dist_attr(name).is_annotated("dims_mapping");
           })
      .def("is_annotated_output_dims_mapping",
           [](const OperatorDistAttr &self, const std::string &name) {
             return self.output_dist_attr(name).is_annotated("dims_mapping");
           })
      .def("rename_input", &OperatorDistAttr::rename_input)
      .def("rename_output", &OperatorDistAttr::rename_output)
      .def("reset", &reset_operator_dist_attr)
      .def("serialize_to_string",
           [](OperatorDistAttr &self) {
             return py::bytes(self.serialize_to_string());
           })
      .def("parse_from_string", &OperatorDistAttr::parse_from_string)
      .def(py::self == py::self)  // NOLINT
      .def(py::self != py::self)  // NOLINT
      .def("__copy__",
           [](const OperatorDistAttr &self) { return OperatorDistAttr(self); })
      .def(
          "__deepcopy__",
          [](const OperatorDistAttr &self, py::dict) {
            return OperatorDistAttr(self);
          },
          py::arg("memo"))
      .def("__str__", &OperatorDistAttr::to_string);

  m->def(
      "contains_spmd_rule",
      [](const std::string op_type) {
        return phi::distributed::SpmdRuleFactory::Instance().ContainsSpmdRule(
            op_type);
      },
      py::return_value_policy::reference);

  m->def(
      "get_phi_spmd_rule",
      [](const std::string op_type) {
        return phi::distributed::SpmdRuleFactory::Instance().GetSpmdRule(
            op_type);
      },
      py::return_value_policy::reference);

  m->def(
      "reshard",
      [](py::handle py_tensor, const TensorDistAttr &dist_attr) {
        auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
        return reshard_ad_function(tensor, dist_attr);
      },
      py::return_value_policy::reference);

  m->def(
      "dtensor_to_local",
      [](py::handle py_tensor) {
        auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
        return dtensor_to_local_ad_function(tensor);
      },
      py::return_value_policy::reference);

  m->def(
      "dtensor_from_local",
      [](py::handle py_tensor,
         py::handle py_process_mesh,
         py::handle py_placements) {
        auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
        auto process_mesh = CastPyArg2ProcessMesh(py_process_mesh.ptr(), 1);
        auto placements = CastPyArg2VectorOfPlacement(py_placements.ptr(), 2);
        return dtensor_from_local_ad_function(tensor, process_mesh, placements);
      },
      py::return_value_policy::reference);

  // TODO(liuzhenhai): DistributedMapper is not used for now, but
  // dist_mapper_test need the symbols touch DistributedMapper to be linked,
  // remove it later
  m->def("touch_dist_mapper", []() {
    DistributedMapper mapper;
    return mapper.to_string();
  });

  m->def("sub_mesh_dim", &phi::distributed::SubMeshDim);
}

static void parse_tensors(PyObject *obj,
                          phi::distributed::InferSpmdContext *ctx,
                          const size_t arg_pos) {
  Py_ssize_t len = PyList_Size(obj);
  VLOG(6) << "args indx: [" << arg_pos << "] input vector of ["
          << static_cast<size_t>(len) << "] tensors.";
  paddle::small_vector<phi::distributed::DistMetaTensor,
                       phi::kInputSmallVectorSize>
      ins;
  ins.reserve(static_cast<size_t>(len));
  for (Py_ssize_t i = 0; i < len; i++) {
    DistTensorSpec in = py::cast<DistTensorSpec>(PyList_GetItem(obj, i));
    VLOG(6) << "Vector emplace_back DistTensorSpec: " << in.to_string();
    ins.emplace_back(phi::distributed::DistMetaTensor(
        common::make_ddim(in.shape()), in.dist_attr()));
  }
  ctx->EmplaceBackInputs(ins);
}

static void parse_tensor(PyObject *obj,
                         phi::distributed::InferSpmdContext *ctx,
                         const size_t arg_pos) {
  VLOG(6) << "args indx: [" << arg_pos << "] input one tensor.";
  DistTensorSpec in = py::cast<DistTensorSpec>(obj);
  VLOG(6) << "DistTensorSpec: " << in.to_string();
  ctx->EmplaceBackInput(phi::distributed::DistMetaTensor(
      common::make_ddim(in.shape()), in.dist_attr()));
}

// TODO(ljz) support other types
static void parse_attrs(PyObject *obj,
                        PyObject *first_item,
                        phi::distributed::InferSpmdContext *ctx,
                        const size_t arg_pos) {
  if (PyBool_Check(first_item)) {
    auto attrs = CastPyArg2Booleans(
        obj, infer_spmd_string, static_cast<ssize_t>(arg_pos));
    ctx->EmplaceBackAttr(attrs);
  } else if (PyCheckInteger(first_item)) {
    auto attrs =
        CastPyArg2Ints(obj, infer_spmd_string, static_cast<ssize_t>(arg_pos));
    ctx->EmplaceBackAttr(attrs);
  } else if (PyLong_Check(first_item)) {
    auto attrs =
        CastPyArg2Longs(obj, infer_spmd_string, static_cast<ssize_t>(arg_pos));
    ctx->EmplaceBackAttr(attrs);
  } else if (PyFloat_Check(first_item)) {
    auto attrs =
        CastPyArg2Floats(obj, infer_spmd_string, static_cast<ssize_t>(arg_pos));
    ctx->EmplaceBackAttr(attrs);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "list of int, float, bool or Tensor, but got %s",
        infer_spmd_string,
        arg_pos,
        ((PyTypeObject *)first_item->ob_type)->tp_name));  // NOLINT
  }
}

// TODO(ljz) support other types
static void parse_attr(PyObject *obj,
                       phi::distributed::InferSpmdContext *ctx,
                       const size_t arg_pos) {
  if (PyBool_Check(obj)) {
    auto attr = CastPyArg2Boolean(
        obj, infer_spmd_string, static_cast<ssize_t>(arg_pos));
    ctx->EmplaceBackAttr(attr);
  } else if (PyCheckInteger(obj)) {
    auto attr =
        CastPyArg2Int(obj, infer_spmd_string, static_cast<ssize_t>(arg_pos));
    ctx->EmplaceBackAttr(attr);
  } else if (PyLong_Check(obj)) {
    auto attr =
        CastPyArg2Long(obj, infer_spmd_string, static_cast<ssize_t>(arg_pos));
    ctx->EmplaceBackAttr(attr);
  } else if (PyFloat_Check(obj)) {
    auto attr =
        CastPyArg2Float(obj, infer_spmd_string, static_cast<ssize_t>(arg_pos));
    ctx->EmplaceBackAttr(attr);
  } else {  // TODO(ljz) support other types
    PADDLE_THROW(common::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "int, float, bool or Tensor, but got %s",
        infer_spmd_string,
        arg_pos,
        ((PyTypeObject *)obj->ob_type)->tp_name));  // NOLINT
  }
}

static void parse_single_pyobject(PyObject *obj,
                                  phi::distributed::InferSpmdContext *ctx,
                                  const size_t arg_pos) {
  if (PyList_Check(obj)) {  // list inputs, spmd not allow tuple inputs
    PyObject *first_item = PyList_GetItem(obj, 0);
    if (PyObject_TypeCheck(first_item, g_dist_tensor_spec_pytype)) {
      parse_tensors(obj, ctx, arg_pos);
    } else {
      parse_attrs(obj, first_item, ctx, arg_pos);
    }
  } else {
    if (PyObject_TypeCheck(obj, g_dist_tensor_spec_pytype)) {
      parse_tensor(obj, ctx, arg_pos);
    } else {
      parse_attr(obj, ctx, arg_pos);
    }
  }
}

static void prepare_ctx(phi::distributed::InferSpmdContext *ctx,
                        const py::args &args) {
  VLOG(6) << "prepare_ctx ";
  size_t inputs_size = args.size();
  for (size_t i = 0; i < inputs_size; ++i) {
    PyObject *obj = args[i].ptr();
    parse_single_pyobject(obj, ctx, i);
  }
}
static std::pair<std::vector<ArgDistAttr>, std::vector<ArgDistAttr>>
infer_forward(const phi::distributed::SpmdRule &self, const py::args &args) {
  VLOG(6) << "infer_forward ";
  phi::distributed::InferSpmdContext ctx;
  prepare_ctx(&ctx, args);
  return self.InferForward(ctx);
}

static std::pair<std::vector<ArgDistAttr>, std::vector<ArgDistAttr>>
infer_backward(const phi::distributed::SpmdRule &self, const py::args &args) {
  VLOG(6) << "infer_backward ";
  phi::distributed::InferSpmdContext ctx;
  prepare_ctx(&ctx, args);
  return self.InferBackward(ctx);
}

}  // namespace paddle::pybind
