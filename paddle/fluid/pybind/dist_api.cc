// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "pybind11/stl.h"

#include "paddle/fluid/distributed/collective/reducer.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_api.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_tools.h"
#include "paddle/fluid/pir/dialect/distributed/transforms/dist_to_dense_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"
#include "paddle/fluid/pybind/dist_api.h"
#include "paddle/fluid/pybind/dist_static_op_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace py = pybind11;

namespace pybind11::detail {
template <typename Key,
          typename Value,
          typename Hash,
          typename Equal,
          typename Alloc>
struct type_caster<paddle::flat_hash_map<Key, Value, Hash, Equal, Alloc>>
    : map_caster<paddle::flat_hash_map<Key, Value, Hash, Equal, Alloc>,
                 Key,
                 Value> {};
}  // namespace pybind11::detail

using paddle::dialect::DistTypeInterface;
using paddle::dialect::OperationDistAttribute;
using paddle::dialect::PlacementsAttribute;
using paddle::dialect::ProcessMeshAttribute;
using paddle::dialect::TensorDistAttribute;
using pir::ArrayAttribute;
using pir::DenseTensorType;
using pir::Value;
using pybind11::return_value_policy;
namespace paddle::pybind {

void BindOperationDistAttribute(py::module *m) {
  py::class_<OperationDistAttribute, pir::Attribute> dist_attr(
      *m, "OperationDistAttribute");
  dist_attr
      .def("__str__",
           [](OperationDistAttribute &self) {
             std::ostringstream print_stream;
             print_stream << self;
             return print_stream.str();
           })
      .def_property_readonly("process_mesh",
                             [](OperationDistAttribute &self) {
                               return self.process_mesh_attr().process_mesh();
                             })
      .def_property_readonly(
          "chunk_id",
          [](OperationDistAttribute &self) { return self.chunk_id(); })
      .def("num_operands", &OperationDistAttribute::num_operands)
      .def("operands", &OperationDistAttribute::operands)
      .def("operand", &OperationDistAttribute::operand)
      .def("num_results", &OperationDistAttribute::num_results)
      .def("results", &OperationDistAttribute::results)
      .def("result", &OperationDistAttribute::result);
}

void BindTensorDistAttribute(py::module *m) {
  py::class_<TensorDistAttribute, pir::Attribute> dist_attr(
      *m, "TensorDistAttribute");
  dist_attr
      .def("__str__",
           [](TensorDistAttribute &self) {
             std::ostringstream print_stream;
             print_stream << self;
             return print_stream.str();
           })
      .def_property_readonly("process_mesh",
                             [](TensorDistAttribute &self) {
                               return self.process_mesh_attr().process_mesh();
                             })
      .def_property_readonly(
          "dims_mapping",
          [](TensorDistAttribute &self) { return self.dims_mapping(); })
      .def_property_readonly(
          "partial_status",
          [](TensorDistAttribute &self) { return self.partial_status(); })
      .def_property_readonly(
          "partial_dims",
          [](TensorDistAttribute &self) { return self.partial_dims(); })
      .def_property_readonly(
          "placements",
          [](TensorDistAttribute &self) { return self.placements(); })
      .def_property_readonly("placements_attr", [](TensorDistAttribute &self) {
        std::optional<PlacementsAttribute> placements_attr =
            self.placements_attr();
        if (placements_attr.has_value()) {
          PyObject *py_obj = ToPyObject(placements_attr->placements());
          return py::reinterpret_borrow<py::object>(py_obj);
        } else {
          return py::none().cast<py::object>();
        }
      });
}

void BindDistType(py::module *m) {
  py::class_<DistTypeInterface, pir::Type> dist_type(*m, "DistType");
  dist_type.def("dist_attr", &DistTypeInterface::tensor_dist_attr)
      .def("local_type", &DistTypeInterface::local_type);
}

void BindDistOpsAPI(pybind11::module *module) {
  {
    if (PyModule_AddFunctions(module->ptr(), DistOpsAPI) < 0) {
      {
        PADDLE_THROW(
            common::errors::Fatal("Add C++ DistOpsAPI to core.ops failed!"));
      }
    }
  }
}

TensorDistAttribute CreateTensorDistAttribute(
    const phi::distributed::ProcessMesh &mesh,
    const std::vector<int64_t> &dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType> &partial_status = {}) {
  return TensorDistAttribute::get(
      pir::IrContext::Instance(), mesh, dims_mapping, partial_status);
}

OperationDistAttribute CreateOperationDistAttribute(
    const phi::distributed::ProcessMesh &mesh,
    const std::vector<pir::Attribute> &operands,
    const std::vector<pir::Attribute> &results,
    const int64_t &chunk_id) {
  return OperationDistAttribute::get(
      pir::IrContext::Instance(), mesh, operands, results, chunk_id);
}

ArrayAttribute CreateArrayAttribute(
    const std::vector<pir::Attribute> &elements) {
  return ArrayAttribute::get(pir::IrContext::Instance(), elements);
}

std::vector<std::vector<size_t>> AssignValueGroupBySize(
    const std::vector<Value> &values,
    const std::vector<size_t> &group_size_limits) {
  std::vector<Tensor> tensors;
  tensors.reserve(values.size());
  for (auto value : values) {
    auto x = value.type().dyn_cast<DenseTensorType>();
    PADDLE_ENFORCE_NOT_NULL(
        x,
        common::errors::Fatal(
            "Only support assign group for dense tensor value!"));
    auto ir_tensor = std::make_shared<dialect::IrTensor>(
        dialect::TransToPhiDataType(x.dtype()),
        x.dims(),
        x.data_layout(),
        x.lod(),
        x.offset());
    tensors.emplace_back(ir_tensor);
  }
  return distributed::Eager_AssignGroupBySize(
      tensors, std::vector<bool>(tensors.size(), false), group_size_limits);
}

void BindDistUtils(pybind11::module *m) {
  m->def("create_tensor_dist_attribute", CreateTensorDistAttribute);
  m->def("create_array_dist_attribute", CreateArrayAttribute);
  m->def("create_op_dist_attribute", CreateOperationDistAttribute);
  m->def("create_op_dist_attribute",
         &CreateOperationDistAttribute,
         py::arg("mesh"),
         py::arg("operands"),
         py::arg("results"),
         py::arg("chunk_id") = -1);
  m->def("create_process_mesh",
         [](const std::vector<int64_t> &shape,
            const std::vector<int64_t> &process_ids,
            const std::vector<std::string> &dim_names) {
           return phi::distributed::ProcessMesh(shape, process_ids, dim_names);
         });
  m->def("create_array_attribute", CreateArrayAttribute);
  m->def("get_sub_meshes", phi::distributed::GetSubMeshes);
  m->def("cvt_to_dist_type",
         &dialect::CvtToPirDistType,
         py::arg("global_type"),
         py::arg("dist_attr"),
         py::arg("local_ddim") = std::vector<int64_t>());
  m->def("assign_value_group_by_size",
         AssignValueGroupBySize,
         py::arg("values"),
         py::arg("group_size_limits"));
}

void BindDistPassAPI(pybind11::module *module) {
  module->def("apply_dist2dense_pass", paddle::dialect::DistToDensePass);
}

void BindOpsFunction(py::module *m) {
  m->def("reshard_v2",
         [](const pir::Value &x, const TensorDistAttribute &dist_attr) {
           return reshard(x, dist_attr);
         });
}

void BindDistApi(pybind11::module *module) {
  auto ir_module = module->def_submodule("pir");
  BindOperationDistAttribute(&ir_module);
  BindTensorDistAttribute(&ir_module);
  BindDistType(&ir_module);
  BindDistUtils(&ir_module);
  BindDistPassAPI(&ir_module);
  auto ops_modules = ir_module.def_submodule("ops");
  BindDistOpsAPI(&ops_modules);
  BindOpsFunction(&ops_modules);
}

}  // namespace paddle::pybind
