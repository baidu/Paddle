// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <memory>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/flags.h"

#ifdef CINN_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "paddle/cinn/backends/cuda_util.h"
#endif

namespace py = pybind11;
namespace cinn::pybind {
namespace {
using py::arg;
void BindCinnRuntime(py::module *);

cinn_type_t NumpyTypeToCinn(py::dtype dt) {
  if (dt.is(py::dtype::of<int32_t>())) {
    return cinn_int32_t();
  } else if (dt.is(py::dtype::of<int64_t>())) {
    return cinn_int64_t();
  } else if (dt.is(py::dtype::of<uint32_t>())) {
    return cinn_uint32_t();
  } else if (dt.is(py::dtype::of<uint64_t>())) {
    return cinn_uint64_t();
  } else if (dt.is(py::dtype::of<float>())) {
    return cinn_float32_t();
  } else if (dt.is(py::dtype::of<double>())) {
    return cinn_float64_t();
  } else if (dt.is(py::dtype::of<bool>())) {
    return cinn_bool_t();
  } else if (dt.is(py::dtype::of<int8_t>())) {
    return cinn_int8_t();
  }

  return cinn_unk_t();
}

cinn_buffer_t *CreateBufferFromNumpy(py::array data,
                                     cinn_device_kind_t device,
                                     int align = 0) {
  cinn_type_t type = NumpyTypeToCinn(data.dtype());
  std::vector<int> shape;
  std::copy_n(data.shape(), data.ndim(), std::back_inserter(shape));
  auto *buffer = cinn_buffer_t::new_(device, type, shape, align);
  cinn_buffer_malloc(nullptr, buffer);
  std::memcpy(buffer->memory, data.data(), data.nbytes());

  return buffer;
}

cinn_buffer_t *CreateBufferFromNumpyImpl(common::UnknownArch, py::array data) {
  LOG(FATAL) << "NotImplemented.";
}

cinn_buffer_t *CreateBufferFromNumpyImpl(common::X86Arch, py::array data) {
  return CreateBufferFromNumpy(data, cinn_x86_device);
}

cinn_buffer_t *CreateBufferFromNumpyImpl(common::ARMArch, py::array data) {
  LOG(FATAL) << "NotImplemented.";
}

cinn_buffer_t *CreateBufferFromNumpyImpl(common::NVGPUArch, py::array data) {
#ifdef CINN_WITH_CUDA
  std::vector<int> shape;
  std::copy_n(data.shape(), data.ndim(), std::back_inserter(shape));
  auto *buffer = new cinn_buffer_t();
  buffer->device = cinn_nvgpu_device;
  buffer->memory_size = data.nbytes();
  CUDA_CALL(cudaMalloc(&buffer->memory, data.nbytes()));
  CUDA_CALL(cudaMemcpy(
      buffer->memory, data.data(), data.nbytes(), cudaMemcpyHostToDevice));
  return buffer;
#else
  PADDLE_THROW(::common::errors::Fatal(
      "To use CUDA backends, you need to set WITH_CUDA ON!"));
#endif
}

cinn_buffer_t *CreateBufferFromNumpyImpl(common::HygonDCUArchHIP,
                                         py::array data) {
  PADDLE_THROW(::common::errors::Unimplemented("CINN old obsolete code!"));
}

cinn_buffer_t *CreateBufferFromNumpyImpl(common::HygonDCUArchSYCL,
                                         py::array data) {
  PADDLE_THROW(::common::errors::Unimplemented("CINN old obsolete code!"));
}

cinn_buffer_t *InterfaceCreateBufferFromNumpy(common::Arch arch,
                                              py::array data) {
  return std::visit(
      [&](const auto &impl) { return CreateBufferFromNumpyImpl(impl, data); },
      arch.variant());
}

cinn_buffer_t *CreateBufferFromNumpy(
    py::array data,
    cinn::common::Target target = cinn::common::DefaultHostTarget(),
    int align = 0) {
  return InterfaceCreateBufferFromNumpy(target.arch, data);
}

void BufferCopyTo(const cinn_buffer_t &buffer, py::array array) {
  void *array_data = array.mutable_data();
  if (buffer.device == cinn_x86_device) {
    std::memcpy(array_data, buffer.memory, array.nbytes());
  } else if (buffer.device == cinn_nvgpu_device) {
#ifdef CINN_WITH_CUDA
    CUDA_CALL(cudaMemcpy(
        array_data, buffer.memory, array.nbytes(), cudaMemcpyDeviceToHost));
#else
    PADDLE_THROW(::common::errors::Fatal(
        "To use CUDA backends, you need to set WITH_CUDA ON!"));
#endif

  } else {
    CINN_NOT_IMPLEMENTED
  }
}

py::array BufferHostMemoryToNumpy(cinn_buffer_t &buffer) {  // NOLINT
  py::dtype dt;
  if (buffer.type == cinn_int32_t()) {
    dt = py::dtype::of<int32_t>();
  } else if (buffer.type == cinn_int64_t()) {
    dt = py::dtype::of<int64_t>();
  } else if (buffer.type == cinn_uint32_t()) {
    dt = py::dtype::of<uint32_t>();
  } else if (buffer.type == cinn_uint64_t()) {
    dt = py::dtype::of<uint64_t>();
  } else if (buffer.type == cinn_float32_t()) {
    dt = py::dtype::of<float>();
  } else if (buffer.type == cinn_float64_t()) {
    dt = py::dtype::of<double>();
  } else if (buffer.type == cinn_int8_t()) {
    dt = py::dtype::of<int8_t>();
  } else if (buffer.type == cinn_bool_t()) {
    dt = py::dtype::of<bool>();
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument("Not supported type found"));
  }

  py::array::ShapeContainer shape(buffer.dims, buffer.dims + buffer.dimensions);
  py::array array(std::move(dt), std::move(shape));
  void *mutable_data = array.mutable_data();
  cinn_buffer_copy_to_host(nullptr, &buffer);
  if (buffer.device == cinn_x86_device) {
    std::memcpy(mutable_data, buffer.memory, buffer.memory_size);
  } else {
    CINN_RUNTIME_NOT_IMPLEMENTED
  }
  return array;
}

struct VoidPointer {
  void *ptr{nullptr};
};

void BindSpecialTypes(py::module *m) {
  py::class_<VoidPointer> void_ptr(*m, "VoidPointer");
  void_ptr.def(py::init<>());

#define VOID_PTR_SUPPORT_TYPE(__type)                    \
  void_ptr.def("set", [](VoidPointer &self, __type *p) { \
    self.ptr = static_cast<void *>(p);                   \
  })

  VOID_PTR_SUPPORT_TYPE(char);
  VOID_PTR_SUPPORT_TYPE(int8_t);
  VOID_PTR_SUPPORT_TYPE(int16_t);
  VOID_PTR_SUPPORT_TYPE(int32_t);
  VOID_PTR_SUPPORT_TYPE(int64_t);
  VOID_PTR_SUPPORT_TYPE(float);
  VOID_PTR_SUPPORT_TYPE(double);
#undef VOID_PTR_SUPPORT_TYPE

  m->def("nullptr", []() { return VoidPointer(); });
}

void BindCinnRuntime(py::module *m) {
  py::enum_<cinn_type_code_t> cinn_type_code(*m, "cinn_type_code_t");
  cinn_type_code.value("cinn_type_unk", cinn_type_unk)
      .value("cinn_type_int", cinn_type_int)
      .value("cinn_type_uint", cinn_type_uint)
      .value("cinn_type_float", cinn_type_float)
      .value("cinn_type_handle", cinn_type_handle)
      .export_values();

  py::class_<cinn_type_t> cinn_type(*m, "cinn_type_t");
  cinn_type.def_readwrite("code", &cinn_type_t::code)
      .def_readwrite("bits", &cinn_type_t::bits)
      .def_readwrite("lanes", &cinn_type_t::lanes)
      .def(py::init<>())
      .def(py::init<cinn_type_code_t, uint8_t, uint16_t>(),
           arg("code"),
           arg("bits"),
           arg("lanes") = 1)
      .def(py::self == cinn_type_t())
      .def(py::self != cinn_type_t())
      .def("bytes", &cinn_type_t::bytes);

  m->def("cinn_unk_t", &cinn_unk_t)
      .def("cinn_int8_t", &cinn_int8_t)
      .def("cinn_bool_t", &cinn_bool_t)
      .def("cinn_int32_t", &cinn_int32_t)
      .def("cinn_int64_t", &cinn_int64_t)
      .def("cinn_uint32_t", &cinn_uint32_t)
      .def("cinn_uint64_t", &cinn_uint64_t)
      .def("cinn_float32_t", &cinn_float32_t)
      .def("cinn_float64_t", &cinn_float64_t);

  py::enum_<cinn_device_kind_t> cinn_device_kind(*m, "cinn_device_kind_t");
  cinn_device_kind.value("cinn_unk_device", cinn_unk_device)
      .value("cinn_x86_device", cinn_x86_device)
      .value("cinn_opencl_device", cinn_opencl_device)
      .value("cinn_arm_device", cinn_arm_device)
      .value("cinn_nvgpu_device", cinn_nvgpu_device)
      .export_values();

  py::enum_<cinn_buffer_kind_t> cinn_buffer_kind(*m, "cinn_buffer_kind_t");
  cinn_buffer_kind.value("cinn_buffer_on_host", cinn_buffer_on_host)
      .value("cinn_buffer_on_device", cinn_buffer_on_device)
      .export_values();

  py::class_<cinn_device_interface_t> cinn_device_interface(
      *m, "cinn_device_interface_t");

  m->def("cinn_device_release", &cinn_device_release);
  m->def("cinn_buffer_copy_to_host", &cinn_buffer_copy_to_host);
  m->def("cinn_buffer_copy_to_device", &cinn_buffer_copy_to_device);
  m->def("cinn_buffer_copy", &cinn_buffer_copy);
  m->def("cinn_device_sync", &cinn_device_sync);
  m->def("cinn_buffer_malloc", &cinn_buffer_malloc);
  m->def("cinn_buffer_malloc", [](VoidPointer &p, cinn_buffer_t *buffer) {
    return cinn_buffer_malloc(p.ptr, buffer);
  });
  m->def("cinn_buffer_free", &cinn_buffer_free);
  m->def("cinn_buffer_get_data_handle", &cinn_buffer_get_data_handle);
  m->def("cinn_buffer_get_data_const_handle",
         &cinn_buffer_get_data_const_handle);

  py::class_<cinn_buffer_t> cinn_buffer(*m, "cinn_buffer_t");
  cinn_buffer.def_readwrite("device", &cinn_buffer_t::device)
      .def_readwrite("device_interface", &cinn_buffer_t::device_interface)
      .def_readwrite("memory", &cinn_buffer_t::memory)
      .def_readwrite("flag", &cinn_buffer_t::flag)
      .def_readwrite("type", &cinn_buffer_t::type)
      .def_readwrite("dimensions", &cinn_buffer_t::dimensions)
      // .def_readwrite("dims", &cinn_buffer_t::dims)
      .def_readwrite("lazy", &cinn_buffer_t::lazy)
      .def_readwrite("memory_size", &cinn_buffer_t::memory_size)
      .def_readwrite("align", &cinn_buffer_t::align)
      .def(py::init<>())
      .def_static("new",
                  &cinn_buffer_t::new_,
                  arg("device"),
                  arg("type"),
                  arg("shape"),
                  arg("align") = 0,
                  py::return_value_policy::reference)
      .def_static("delete", &cinn_buffer_t::delete_)
      // .def_static("alloc", &cinn_buffer_t::alloc)
      .def("resize", &cinn_buffer_t::resize)
      .def("num_elements", &cinn_buffer_t::num_elements)
      .def("on_host", &cinn_buffer_t::on_host)
      .def("on_device", &cinn_buffer_t::on_device)
      .def("set_on_host", &cinn_buffer_t::set_on_host, arg("x") = true)
      .def("set_on_device", &cinn_buffer_t::set_on_device, arg("x") = true)
      .def("device_sync", &cinn_buffer_t::device_sync, arg("ctx") = nullptr)
      .def("begin", &cinn_buffer_t::begin, py::return_value_policy::reference)
      .def("end", &cinn_buffer_t::end, py::return_value_policy::reference)
      .def("get_flag", &cinn_buffer_t::get_flag)
      .def("set_flag", &cinn_buffer_t::set_flag)
      // Python methods
      .def("numpy", &BufferHostMemoryToNumpy)
      .def(py::init(py::overload_cast<py::array, cinn_device_kind_t, int>(
               &CreateBufferFromNumpy)),
           arg("data"),
           arg("device"),
           arg("align") = 0)
      .def(py::init(py::overload_cast<py::array, cinn::common::Target, int>(
               &CreateBufferFromNumpy)),
           arg("data"),
           arg("target"),
           arg("align") = 0)
      .def("copy_to", &BufferCopyTo);

  m->def("cinn_x86_device_interface", &cinn_x86_device_interface)
      .def("cinn_buffer_load_float32", &cinn_buffer_load_float32)
      .def("cinn_buffer_load_float64", &cinn_buffer_load_float64);
  // .def("cinn_buffer_slice", &cinn_buffer_slice,
  //     py::return_value_policy::reference);

  py::class_<cinn_value_t> cinn_value(*m, "cinn_value_t");
  cinn_value.def(py::init<>())
      .def_property(
          "v_int64",
          [](cinn_value_t &self) -> const int64_t { return self.v_int64; },
          [](cinn_value_t &self, int64_t v) { self.v_int64 = v; })
      .def_property(
          "v_float64",
          [](cinn_value_t &self) -> const double { return self.v_float64; },
          [](cinn_value_t &self, double v) { self.v_float64 = v; })
      .def_property(
          "v_handle",
          [](cinn_value_t &self) -> const void * { return self.v_handle; },
          [](cinn_value_t &self, void *v) { self.v_handle = v; })
      .def_property(
          "v_str",
          [](cinn_value_t &self) -> const char * { return self.v_str; },
          [](cinn_value_t &self, char *v) { self.v_str = v; });
  py::class_<cinn_pod_value_t> cinn_pod_value(*m, "cinn_pod_value_t");
  cinn_pod_value.def(py::init<>())
      .def(py::init<cinn_value_t, int>())
      .def(py::init<cinn_buffer_t *>())
      .def(py::init<bool>())
      .def(py::init<int8_t>())
      .def(py::init<int32_t>())
      .def(py::init<int64_t>())
      .def(py::init<float>())
      .def(py::init<double>())
      .def(py::init<void *>())
      .def(py::init<const char *>())
      .def("to_double", &cinn_pod_value_t::operator double)
      .def("to_float", &cinn_pod_value_t::operator float)
      .def("to_int8", &cinn_pod_value_t::operator int8_t)
      .def("to_int32", &cinn_pod_value_t::operator int32_t)
      .def("to_int64", &cinn_pod_value_t::operator int64_t)
      .def("to_void_p", &cinn_pod_value_t::operator void *)
      .def("to_cinn_buffer_t_p", &cinn_pod_value_t::operator cinn_buffer_t *)
      .def("to_char_p", &cinn_pod_value_t::operator char *)
      .def("type_code",
           py::overload_cast<>(&cinn_pod_value_t::type_code, py::const_));

  m->def("set_cinn_cudnn_deterministic",
         &cinn::runtime::SetCinnCudnnDeterministic,
         py::arg("state") = true);
  m->def("seed", &cinn::runtime::RandomSeed::GetOrSet, py::arg("seed") = 0);
  m->def("clear_seed", &cinn::runtime::RandomSeed::Clear);
}
}  // namespace

void BindRuntime(py::module *m) {
  BindSpecialTypes(m);
  BindCinnRuntime(m);
}
}  // namespace cinn::pybind
