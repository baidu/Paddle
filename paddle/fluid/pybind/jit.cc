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

#include "paddle/fluid/pybind/jit.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/fluid/pybind/sot/eval_frame.h"
#include "paddle/fluid/pybind/sot/eval_frame_tools.h"
#include "paddle/fluid/pybind/sot/frame_proxy.h"
#include "paddle/fluid/pybind/sot/guards.h"
#include "paddle/fluid/pybind/sot/macros.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;

namespace paddle::pybind {

PyTypeObject *g_jit_function_pytype = nullptr;
using Variable = paddle::framework::Variable;

void BindJit(pybind11::module *m) {
  py::class_<jit::Layer>(*m, "Layer", R"DOC(Layer Class.)DOC")
      .def("function_names", &jit::Layer::FunctionNames)
      .def("function", &jit::Layer::Function)
      .def("function_info", &jit::Layer::FunctionInfo);

  py::class_<jit::Function, std::shared_ptr<jit::Function>> function(
      *m, "Function", R"DOC(Function Class.)DOC");
  g_jit_function_pytype = reinterpret_cast<PyTypeObject *>(function.ptr());

  py::class_<jit::BaseFunctionInfo, std::shared_ptr<jit::BaseFunctionInfo>>(
      *m, "FunctionInfo", R"DOC(BaseFunctionInfo Class.)DOC")
      .def("name", &jit::BaseFunctionInfo::FunctionName)
      .def("input_names", &jit::BaseFunctionInfo::InputArgNames)
      .def("output_names", &jit::BaseFunctionInfo::OutputArgNames);

  m->def("Load", [](const std::string &path, const phi::CPUPlace &cpu_place) {
    return paddle::jit::Load(path, cpu_place);
  });

  m->def("Load", [](const std::string &path, const phi::GPUPlace &cuda_place) {
    return paddle::jit::Load(path, cuda_place);
  });
}

void BindGuard(pybind11::module *m) {
#if SOT_IS_SUPPORTED
  py::class_<GuardBase, std::shared_ptr<GuardBase>>(
      *m, "GuardBase", R"DOC(GuardBase Class.)DOC")
      .def("check", &GuardBase::check_pybind);
  py::class_<LambdaGuard, GuardBase, std::shared_ptr<LambdaGuard>>(
      *m, "LambdaGuard", R"DOC(LambdaGuard Class.)DOC")
      .def(py::init<const py::function &>(), py::arg("guard_check_fn"));
  py::class_<GuardGroup, GuardBase, std::shared_ptr<GuardGroup>>(
      *m, "GuardGroup", R"DOC(GuardGroup Class.)DOC")
      .def(py::init<const std::vector<std::shared_ptr<GuardBase>> &>(),
           py::arg("guards"));
  py::class_<TypeMatchGuard, GuardBase, std::shared_ptr<TypeMatchGuard>>(
      *m, "TypeMatchGuard", R"DOC(TypeMatchGuard Class.)DOC")
      .def(py::init<const py::type &>(), py::arg("py_type"));
  py::class_<IdMatchGuard, GuardBase, std::shared_ptr<IdMatchGuard>>(
      *m, "IdMatchGuard", R"DOC(IdMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("py_obj"));
  py::class_<LengthMatchGuard, GuardBase, std::shared_ptr<LengthMatchGuard>>(
      *m, "LengthMatchGuard", R"DOC(LengthMatchGuard Class.)DOC")
      .def(py::init<const Py_ssize_t &>(), py::arg("length"));
  py::class_<ValueMatchGuard, GuardBase, std::shared_ptr<ValueMatchGuard>>(
      *m, "ValueMatchGuard", R"DOC(ValueMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("py_value"));
  py::class_<DtypeMatchGuard, GuardBase, std::shared_ptr<DtypeMatchGuard>>(
      *m, "DtypeMatchGuard", R"DOC(DtypeMatchGuard Class.)DOC")
      .def(py::init<const paddle::framework::proto::VarType &>(),
           py::arg("dtype"))
      .def(py::init<const phi::DataType &>(), py::arg("dtype"));
  py::class_<AttributeMatchGuard,
             GuardBase,
             std::shared_ptr<AttributeMatchGuard>>(
      *m, "AttributeMatchGuard", R"DOC(AttributeMatchGuard Class.)DOC")
      .def(py::init<const py::object &, const std::string &>(),
           py::arg("obj"),
           py::arg("attr_name"));
  py::class_<ShapeMatchGuard, GuardBase, std::shared_ptr<ShapeMatchGuard>>(
      *m, "ShapeMatchGuard", R"DOC(ShapeMatchGuard Class.)DOC")
      .def(py::init<const std::vector<py::object> &>(), py::arg("shape"));
  py::class_<LayerMatchGuard, GuardBase, std::shared_ptr<LayerMatchGuard>>(
      *m, "LayerMatchGuard", R"DOC(LayerMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("layer_obj"));
  py::class_<RangeMatchGuard, GuardBase, std::shared_ptr<RangeMatchGuard>>(
      *m, "RangeMatchGuard", R"DOC(RangeMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("range_obj"));
  py::class_<InstanceCheckGuard,
             GuardBase,
             std::shared_ptr<InstanceCheckGuard>>(
      *m, "InstanceCheckGuard", R"DOC(InstanceCheckGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("isinstance_obj"));

  m->def(
      "merge_guard",
      [](const std::vector<std::shared_ptr<GuardBase>> &py_guards) {
        return GuardGroup(py_guards);
      },
      py::arg("py_guards"));
#endif
}

void BindSot(pybind11::module *m) {
#if SOT_IS_SUPPORTED
  PyInit__eval_frame();
#if PY_3_11_PLUS
  PyInit__frame_proxy();
#endif
  m->def(
      "set_eval_frame",
      [](const py::object &py_func) {
        VLOG(5) << "start call set_eval_frame_py.";
        auto ret = set_eval_frame_py(py_func.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("callback"));

  m->def("has_custom_getattro", [](py::object obj) {
    PyObject *py_obj = obj.ptr();

    if (!PyType_Check(py_obj)) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The input object should be a type object, but got %s.",
          py::str(py_obj).cast<std::string>()));
    }
    PyTypeObject *type = reinterpret_cast<PyTypeObject *>(py_obj);

    return type->tp_getattro != PyObject_GenericGetAttr;
  });

  m->def(
      "sot_setup_codes_with_graph",
      [](const py::object &py_codes) {
        auto ret = setup_codes_with_graph(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "sot_set_with_graph",
      [](const py::object &py_codes) {
        auto ret = set_with_graph(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "eval_frame_no_skip_codes",
      [](const py::object &py_codes) {
        auto ret = no_skip_codes(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "eval_frame_skip_file_prefix",
      [](const py::object &py_codes) {
        auto ret = skip_file_prefix(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));
  BindGuard(m);
#endif
}

}  // namespace paddle::pybind
