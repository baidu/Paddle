// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/global_value_getter_setter.h"

#include <cctype>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/common/macros.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/platform/enforce.h"
#include "pybind11/stl.h"

// FIXME(zengjinle): these 2 flags may be removed by the linker when compiling
// CPU-only Paddle. It is because they are only used in
// AutoGrowthBestFitAllocator, but AutoGrowthBestFitAllocator is not used
// (in the translation unit level) when compiling CPU-only Paddle. I do not
// want to add PADDLE_FORCE_LINK_FLAG, but I have not found any other methods
// to solve this problem.
PADDLE_FORCE_LINK_FLAG(free_idle_chunk);
PADDLE_FORCE_LINK_FLAG(free_when_no_cache_hit);

// NOTE: where are these 2 flags from?
#ifdef PADDLE_WITH_DISTRIBUTE
PD_DECLARE_int32(rpc_get_thread_num);
PD_DECLARE_int32(rpc_prefetch_thread_num);
#endif

namespace paddle::pybind {

namespace py = pybind11;

class PYBIND11_HIDDEN GlobalVarGetterSetterRegistry {
  DISABLE_COPY_AND_ASSIGN(GlobalVarGetterSetterRegistry);

  GlobalVarGetterSetterRegistry() = default;

 public:
  using Getter = std::function<py::object()>;
  using Setter = std::function<void(const py::object &)>;

  template <typename T>
  static Getter CreateGetter(const T &var) {
    return [&]() -> py::object { return py::cast(var); };
  }

  template <typename T>
  static Getter CreateDefaultValueGetter(const T &var) {
    return [=]() -> py::object { return py::cast(var); };
  }

  template <typename T>
  static Setter CreateSetter(T *var) {
    return [var](const py::object &obj) { *var = py::cast<T>(obj); };
  }

 private:
  struct VarInfo {
    VarInfo(bool is_public, Getter getter, Getter default_getter)
        : is_public(is_public),
          getter(std::move(getter)),
          default_getter(std::move(default_getter)) {}

    VarInfo(bool is_public, Getter getter, Getter default_getter, Setter setter)
        : is_public(is_public),
          getter(std::move(getter)),
          default_getter(std::move(default_getter)),
          setter(std::move(setter)) {}

    const bool is_public;
    const Getter getter;
    const Getter default_getter;
    const Setter setter;
  };

 public:
  static const GlobalVarGetterSetterRegistry &Instance() { return instance_; }

  static GlobalVarGetterSetterRegistry *MutableInstance() { return &instance_; }

  void Register(const std::string &name,
                bool is_public,
                const Getter &getter,
                const Getter &default_getter) {
    PADDLE_ENFORCE_EQ(
        HasGetterMethod(name),
        false,
        common::errors::AlreadyExists(
            "Getter of global variable %s has been registered", name));
    PADDLE_ENFORCE_NOT_NULL(getter,
                            common::errors::InvalidArgument(
                                "Getter of %s should not be null", name));
    var_infos_.insert({name, VarInfo(is_public, getter, default_getter)});
  }

  void Register(const std::string &name,
                bool is_public,
                const Getter &getter,
                const Getter &default_getter,
                const Setter &setter) {
    PADDLE_ENFORCE_EQ(
        HasGetterMethod(name),
        false,
        common::errors::AlreadyExists(
            "Getter of global variable %s has been registered", name));

    PADDLE_ENFORCE_EQ(
        HasSetterMethod(name),
        false,
        common::errors::AlreadyExists(
            "Setter of global variable %s has been registered", name));

    PADDLE_ENFORCE_NOT_NULL(getter,
                            common::errors::InvalidArgument(
                                "Getter of %s should not be null", name));

    PADDLE_ENFORCE_NOT_NULL(setter,
                            common::errors::InvalidArgument(
                                "Setter of %s should not be null", name));
    var_infos_.insert(
        {name, VarInfo(is_public, getter, default_getter, setter)});
  }

  const Getter &GetterMethod(const std::string &name) const {
    PADDLE_ENFORCE_EQ(
        HasGetterMethod(name),
        true,
        common::errors::NotFound("Cannot find global variable %s", name));
    return var_infos_.at(name).getter;
  }

  const Getter &DefaultGetterMethod(const std::string &name) const {
    PADDLE_ENFORCE_EQ(
        HasGetterMethod(name),
        true,
        common::errors::NotFound("Cannot find global variable %s", name));
    return var_infos_.at(name).default_getter;
  }

  py::object GetOrReturnDefaultValue(const std::string &name,
                                     const py::object &default_value) const {
    if (HasGetterMethod(name)) {
      return GetterMethod(name)();
    } else {
      return default_value;
    }
  }

  py::object GetDefaultValue(const std::string &name) const {
    if (HasGetterMethod(name)) {
      return DefaultGetterMethod(name)();
    } else {
      return py::cast(Py_None);
    }
  }

  py::object Get(const std::string &name) const { return GetterMethod(name)(); }

  const Setter &SetterMethod(const std::string &name) const {
    PADDLE_ENFORCE_EQ(
        HasSetterMethod(name),
        true,
        common::errors::NotFound("Global variable %s is not writable", name));
    return var_infos_.at(name).setter;
  }

  void Set(const std::string &name, const py::object &value) const {
    VLOG(4) << "set " << name << " to " << value;
    SetterMethod(name)(value);
  }

  bool HasGetterMethod(const std::string &name) const {
    return var_infos_.count(name) > 0;
  }

  bool HasSetterMethod(const std::string &name) const {
    return var_infos_.count(name) > 0 && var_infos_.at(name).setter;
  }

  bool IsPublic(const std::string &name) const {
    return var_infos_.count(name) > 0 && var_infos_.at(name).is_public;
  }

  std::unordered_set<std::string> Keys() const {
    std::unordered_set<std::string> keys;
    keys.reserve(var_infos_.size());
    for (auto &pair : var_infos_) {
      keys.insert(pair.first);
    }
    return keys;
  }

 private:
  static GlobalVarGetterSetterRegistry instance_;

  std::unordered_map<std::string, VarInfo> var_infos_;
};

GlobalVarGetterSetterRegistry GlobalVarGetterSetterRegistry::instance_;

static void RegisterGlobalVarGetterSetter();

void BindGlobalValueGetterSetter(pybind11::module *module) {
  RegisterGlobalVarGetterSetter();

  py::class_<GlobalVarGetterSetterRegistry>(*module,
                                            "GlobalVarGetterSetterRegistry")
      .def("__getitem__", &GlobalVarGetterSetterRegistry::Get)
      .def("__setitem__", &GlobalVarGetterSetterRegistry::Set)
      .def("__contains__", &GlobalVarGetterSetterRegistry::HasGetterMethod)
      .def("keys", &GlobalVarGetterSetterRegistry::Keys)
      .def("is_public", &GlobalVarGetterSetterRegistry::IsPublic)
      .def("get_default",
           &GlobalVarGetterSetterRegistry::GetDefaultValue,
           py::arg("key"))
      .def("get",
           &GlobalVarGetterSetterRegistry::GetOrReturnDefaultValue,
           py::arg("key"),
           py::arg("default") = py::cast<py::none>(Py_None));

  module->def("globals",
              &GlobalVarGetterSetterRegistry::Instance,
              py::return_value_policy::reference);
}

/* Public vars are designed to be writable. */
#define REGISTER_PUBLIC_GLOBAL_VAR(var)                                \
  do {                                                                 \
    auto *instance = GlobalVarGetterSetterRegistry::MutableInstance(); \
    instance->Register(                                                \
        #var,                                                          \
        /*is_public=*/true,                                            \
        GlobalVarGetterSetterRegistry::CreateGetter(var),              \
        GlobalVarGetterSetterRegistry::CreateDefaultValueGetter(var),  \
        GlobalVarGetterSetterRegistry::CreateSetter(&var));            \
  } while (0)

struct RegisterGetterSetterVisitor {
  RegisterGetterSetterVisitor(std::string name,
                              bool is_writable,
                              void *value_ptr)
      : name_(std::move(name)),
        is_writable_(is_writable),
        value_ptr_(value_ptr) {}

  template <typename T>
  void operator()(const T &default_value) const {
    auto &value = *static_cast<T *>(value_ptr_);
    auto *instance = GlobalVarGetterSetterRegistry::MutableInstance();
    bool is_public = is_writable_;  // currently, all writable vars are public
    if (is_writable_) {
      instance->Register(
          name_,
          is_public,
          GlobalVarGetterSetterRegistry::CreateGetter(value),
          GlobalVarGetterSetterRegistry::CreateDefaultValueGetter(
              default_value),
          GlobalVarGetterSetterRegistry::CreateSetter(&value));
    } else {
      instance->Register(
          name_,
          is_public,
          GlobalVarGetterSetterRegistry::CreateGetter(value),
          GlobalVarGetterSetterRegistry::CreateDefaultValueGetter(
              default_value));
    }
  }

 private:
  std::string name_;
  bool is_writable_;
  void *value_ptr_;
};

static void RegisterGlobalVarGetterSetter() {
  const auto &flag_map = phi::GetExportedFlagInfoMap();
  for (const auto &pair : flag_map) {
    const std::string &name = pair.second.name;
    bool is_writable = pair.second.is_writable;
    void *value_ptr = pair.second.value_ptr;
    const auto &default_value = pair.second.default_value;
    RegisterGetterSetterVisitor visitor(
        "FLAGS_" + name, is_writable, value_ptr);
    paddle::visit(visitor, default_value);
  }
}

}  // namespace paddle::pybind
