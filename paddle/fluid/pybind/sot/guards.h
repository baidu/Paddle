/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#include <Python.h>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/pybind/sot/macros.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/utils/pybind.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;
#define PYBIND11_DETAILED_ERROR_MESSAGES
#if SOT_IS_SUPPORTED

class GuardBase {
 public:
  GuardBase() = default;

  bool check_pybind(py::handle value) { return check(value.ptr()); }

  virtual bool check(PyObject* value) = 0;
  virtual ~GuardBase() = default;
};

class LambdaGuard : public GuardBase {
 public:
  explicit LambdaGuard(PyObject* guard_check_fn)
      : guard_check_fn_(guard_check_fn) {}

  explicit LambdaGuard(const py::function& guard_check_fn)
      : guard_check_fn_(guard_check_fn.ptr()) {
    Py_INCREF(guard_check_fn_);
  }

  ~LambdaGuard() { Py_DECREF(guard_check_fn_); }

  bool check(PyObject* value);

 private:
  PyObject* guard_check_fn_;
};

class GuardGroup : public GuardBase {
 public:
  explicit GuardGroup(const std::vector<std::shared_ptr<GuardBase>>& guards) {
    for (auto& guard : guards) {
      if (auto group = dynamic_cast<GuardGroup*>(guard.get())) {
        guards_.insert(
            guards_.end(), group->guards_.begin(), group->guards_.end());
      } else {
        guards_.push_back(std::move(guard));
      }
    }
  }
  bool check(PyObject* value);

 private:
  std::vector<std::shared_ptr<GuardBase>> guards_;
};

class TypeMatchGuard : public GuardBase {
 public:
  explicit TypeMatchGuard(PyTypeObject* type_ptr) : expected_(type_ptr) {}
  explicit TypeMatchGuard(PyObject* type_ptr)
      : expected_(reinterpret_cast<PyTypeObject*>(type_ptr)) {}
  explicit TypeMatchGuard(const py::type& py_type)
      : expected_(reinterpret_cast<PyTypeObject*>(py_type.ptr())) {}

  bool check(PyObject* value);

 private:
  PyTypeObject* expected_;
};

class IdMatchGuard : public GuardBase {
 public:
  explicit IdMatchGuard(PyObject* obj_ptr)
      : expected_(reinterpret_cast<PyObject*>(obj_ptr)) {}
  explicit IdMatchGuard(const py::object& py_obj)
      : expected_(reinterpret_cast<PyObject*>(py_obj.ptr())) {}

  bool check(PyObject* value);

 private:
  PyObject* expected_;
};

class ValueMatchGuard : public GuardBase {
 public:
  explicit ValueMatchGuard(PyObject* value_ptr)
      : expected_value_(value_ptr), expected_type_(value_ptr->ob_type) {}

  explicit ValueMatchGuard(const py::object& py_value)
      : expected_value_(py_value.ptr()),
        expected_type_(Py_TYPE(py_value.ptr())) {
    Py_INCREF(expected_value_);
  }

  ~ValueMatchGuard() { Py_DECREF(expected_value_); }

  bool check(PyObject* value);

 private:
  PyObject* expected_value_;
  PyTypeObject* expected_type_;
};

class LengthMatchGuard : public GuardBase {
 public:
  explicit LengthMatchGuard(const Py_ssize_t& length) : expected_(length) {}

  bool check(PyObject* value);

 private:
  Py_ssize_t expected_;
};

class DtypeMatchGuard : public GuardBase {
 public:
  explicit DtypeMatchGuard(const paddle::framework::proto::VarType& dtype_ptr)
      : expected_(dtype_ptr.type()) {}

  explicit DtypeMatchGuard(const phi::DataType& dtype_ptr)
      : expected_(phi::TransToProtoVarType(dtype_ptr)) {}

  bool check(PyObject* value);

 private:
  int expected_;
};

class ShapeMatchGuard : public GuardBase {
 public:
  explicit ShapeMatchGuard(const std::vector<std::optional<int64_t>>& shape)
      : expected_(shape) {}

  explicit ShapeMatchGuard(const std::vector<py::object>& shape) {
    expected_.resize(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      if (py::isinstance<py::int_>(shape[i]) && shape[i].cast<int64_t>() > 0) {
        expected_[i] = std::make_optional(shape[i].cast<int64_t>());
      }
    }
  }

  bool check(PyObject* value);

 private:
  std::vector<std::optional<int64_t>> expected_;
};

class AttributeMatchGuard : public GuardBase {
 public:
  AttributeMatchGuard(const py::object& obj, const std::string& attr_name)
      : attr_ptr_(PyObject_GetAttrString(obj.ptr(), attr_name.c_str())),
        attr_name_(attr_name) {}

  bool check(PyObject* value);

 private:
  PyObject* attr_ptr_;
  std::string attr_name_;
};

class LayerMatchGuard : public GuardBase {
 public:
  explicit LayerMatchGuard(PyObject* layer_ptr) : layer_ptr_(layer_ptr) {
    training_ = PyObject_GetAttrString(layer_ptr, "training") == Py_True;
  }

  explicit LayerMatchGuard(const py::object& layer_obj)
      : layer_ptr_(layer_obj.ptr()), training_(layer_obj.attr("training")) {}

  bool check(PyObject* value);

 private:
  PyObject* layer_ptr_;
  bool training_;
};

class RangeMatchGuard : public GuardGroup {
 public:
  explicit RangeMatchGuard(const py::object& range_obj)
      : GuardGroup({std::make_shared<TypeMatchGuard>(Py_TYPE(range_obj.ptr())),
                    std::make_shared<AttributeMatchGuard>(range_obj, "start"),
                    std::make_shared<AttributeMatchGuard>(range_obj, "stop"),
                    std::make_shared<AttributeMatchGuard>(range_obj, "step")}) {
  }
};

class InstanceCheckGuard : public GuardBase {
 public:
  explicit InstanceCheckGuard(const py::object& py_type)
      : expected_(py_type.ptr()) {
    Py_INCREF(expected_);
  }

  ~InstanceCheckGuard() override { Py_DECREF(expected_); }

  bool check(PyObject* value) override;

 private:
  PyObject* expected_;
};

#endif
