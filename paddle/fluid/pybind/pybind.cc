/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <Python.h>
#include "paddle/fluid/eager/grad_node_info.h"

// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>  // NOLINT // for call_once
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/compiled_program.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/dense_tensor_array.h"
#include "paddle/fluid/framework/details/nan_inf_utils_detail.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/ir/coalesce_grad_tensor_pass.h"
#include "paddle/fluid/framework/ir/cost_model.h"
#include "paddle/fluid/framework/ir/generate_pass.h"
#include "paddle/fluid/framework/ir/pass_builder.h"
#include "paddle/fluid/framework/new_executor/collect_shape_manager.h"
#include "paddle/fluid/framework/new_executor/executor_statistics.h"
#include "paddle/fluid/framework/new_executor/interpreter/job.h"
#include "paddle/fluid/framework/new_executor/interpreter/plan.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/prune.h"
#include "paddle/fluid/framework/scope_pool.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/framework/reader.h"
#include "paddle/phi/core/memory/allocation/allocator_strategy.h"
#include "paddle/phi/core/raw_tensor.h"
#include "paddle/phi/core/tensor_meta.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/memory/allocation/auto_growth_best_fit_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/cuda_ipc_allocator.h"
#endif
#include "paddle/common/macros.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/ops_extra_info.h"
#include "paddle/fluid/operators/py_func_op.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/profiler/event_python.h"
#include "paddle/fluid/platform/profiler/profiler.h"
#include "paddle/fluid/platform/tensorrt/engine_params.h"
#include "paddle/fluid/pybind/auto_parallel_py.h"
#include "paddle/fluid/pybind/bind_cost_model.h"
#include "paddle/fluid/pybind/box_helper_py.h"
#include "paddle/fluid/pybind/communication.h"
#include "paddle/fluid/pybind/compatible.h"
#include "paddle/fluid/pybind/const_value.h"
#include "paddle/fluid/pybind/cuda_streams_py.h"
#include "paddle/fluid/pybind/custom_device_py.h"
#include "paddle/fluid/pybind/data_set_py.h"
#include "paddle/fluid/pybind/distributed_py.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/fleet_wrapper_py.h"
#include "paddle/fluid/pybind/generator_py.h"
#include "paddle/fluid/pybind/global_value_getter_setter.h"
#include "paddle/fluid/pybind/gloo_context_py.h"
#include "paddle/fluid/pybind/gloo_wrapper_py.h"
#include "paddle/fluid/pybind/graph.h"
#include "paddle/fluid/pybind/heter_wrapper_py.h"
#include "paddle/fluid/pybind/imperative.h"
#include "paddle/fluid/pybind/inference_api.h"
#include "paddle/fluid/pybind/io.h"
#include "paddle/fluid/pybind/jit.h"
#include "paddle/fluid/pybind/metrics_py.h"
#include "paddle/fluid/pybind/pir.h"
#include "paddle/fluid/pybind/ps_gpu_wrapper_py.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/fluid/pybind/python_callable_registry.h"
#include "paddle/fluid/pybind/xpu_streams_py.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/lod_utils.h"
#include "paddle/phi/core/memory/allocation/mmap_allocator.h"
#include "paddle/phi/core/platform/cpu_helper.h"
#include "paddle/phi/core/platform/device/device_wrapper.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/platform/monitor.h"
#include "paddle/phi/core/platform/profiler.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/utils/none.h"

#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/fluid/pybind/dist_api.h"
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/pybind/nccl_wrapper_py.h"
#endif
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/pybind/compiled_program.h"
#include "paddle/fluid/pybind/place.h"
#include "paddle/fluid/pybind/protobuf.h"
#include "paddle/fluid/pybind/pybind.h"  // NOLINT
#include "paddle/fluid/pybind/reader_py.h"
#include "paddle/fluid/pybind/tensor.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/utils/string/to_string.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#endif
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/gpu/cuda/gpu_event_timer.h"
#endif
#ifndef PADDLE_WITH_HIP
#include "paddle/phi/core/platform/device/gpu/cuda/cuda_profiler.h"
#endif
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#include "paddle/phi/core/platform/device/xpu/xpu_op_list.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/fluid/operators/custom_device_common_op_registry.h"
#include "paddle/fluid/platform/profiler/custom_device/custom_tracer.h"
#include "paddle/phi/capi/capi.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/platform/device/custom/custom_device_resource_pool.h"
#endif

#include "paddle/phi/core/platform/cuda_graph_with_memory_pool.h"

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"
#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#endif

#ifdef PADDLE_WITH_CRYPTO
#include "paddle/fluid/pybind/crypto.h"
#endif

#if defined PADDLE_WITH_PSCORE
#include "paddle/fluid/pybind/fleet_py.h"
#endif

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/pybind/bind.h"
#include "paddle/fluid/pybind/test.h"
#endif

#if defined(PADDLE_WITH_RPC)
#include "paddle/fluid/pybind/rpc.h"
#endif

#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_interface.h"
#include "paddle/fluid/pir/dialect/operator/interface/decomp.h"
#include "paddle/fluid/pir/dialect/operator/interface/decomp_vjp.h"
#include "paddle/fluid/pir/dialect/operator/interface/vjp.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/custom_vjp.h"
#include "paddle/fluid/pir/dialect/operator/trait/forward_only.h"
#include "paddle/fluid/prim/utils/eager/eager_tensor_operants.h"
#include "paddle/fluid/prim/utils/static/static_tensor_operants.h"
#include "paddle/fluid/primitive/base/decomp_trans.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/api/include/operants_manager.h"
#include "paddle/phi/api/include/tensor_operants.h"
#include "paddle/phi/common/type_promotion.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"
#include "pybind11/stl.h"
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/pir/declare_plugin.h"
#include "paddle/fluid/platform/tensorrt/trt_plugin.h"
#endif
#include "paddle/fluid/eager/accumulation/accumulation_node.h"

COMMON_DECLARE_bool(use_mkldnn);
COMMON_DECLARE_string(prim_backward_blacklist);

// disable auto conversion to list in Python
PYBIND11_MAKE_OPAQUE(phi::TensorArray);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchUnmergedList);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchList);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchType);

DECLARE_FILE_SYMBOLS(init_phi);
DECLARE_FILE_SYMBOLS(kernel_dialect);
#ifdef PADDLE_WITH_DISTRIBUTE
DECLARE_FILE_SYMBOLS(dist_dialect);
#endif
DECLARE_FILE_SYMBOLS(buffered_allocator);
DECLARE_FILE_SYMBOLS(best_fit_allocator);
DECLARE_FILE_SYMBOLS(aligned_allocator);
DECLARE_FILE_SYMBOLS(pass_timing);
DECLARE_FILE_SYMBOLS(op_compatible_info);
DECLARE_FILE_SYMBOLS(sub_graph_detector);
DECLARE_FILE_SYMBOLS(pd_op_to_kernel_pass);
namespace paddle::pybind {

PyTypeObject *g_framework_scope_pytype = nullptr;
PyTypeObject *g_framework_densetensorarray_pytype = nullptr;
PyTypeObject *g_custom_op_kernel_ctx_pytype = nullptr;
PyTypeObject *g_data_type_pytype = nullptr;
PyTypeObject *g_tensorrt_engine_params_pytype = nullptr;

bool IsCompiledWithAVX() {
#ifndef PADDLE_WITH_AVX
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithCUDA() {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithCudnnFrontend() {
#ifndef PADDLE_WITH_CUDNN_FRONTEND
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithDISTRIBUTE() {
#if !defined(PADDLE_WITH_DISTRIBUTE)
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithNCCL() {
#ifdef PADDLE_WITH_NCCL
  return true;
#else
  return false;
#endif
}

bool IsCompiledWithMPI() {
#ifdef PADDLE_WITH_MPI
  return true;
#else
  return false;
#endif
}

// NOTE some mpi lib can support cuda aware, support it in the future.
bool IsCompiledWithMPIAWARE() {
#ifdef PADDLE_WITH_MPI_AWARE
  return true;
#else
  return false;
#endif
}

bool IsCompiledWithROCM() {
#ifndef PADDLE_WITH_HIP
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithXPU() {
#ifndef PADDLE_WITH_XPU
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithCustomDevice(std::string device_type) {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
  return false;
#else
  std::vector<std::string> device_types;
  device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (std::count(device_types.begin(), device_types.end(), device_type)) {
    return true;
  } else {
    return false;
  }
#endif
}

bool IsCompiledWithIPU() {
#ifndef PADDLE_WITH_IPU
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithMKLDNN() {
#ifndef PADDLE_WITH_DNNL
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithCINN() {
#ifndef PADDLE_WITH_CINN
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithHETERPS() {
#ifndef PADDLE_WITH_HETERPS
  return false;
#else
  return true;
#endif
}

bool SupportsBfloat16() {
#ifndef PADDLE_WITH_DNNL
  return false;
#else
  if (phi::backends::cpu::MayIUse(phi::backends::cpu::cpu_isa_t::avx512_core))
    return true;
  else
    return false;
#endif
}

bool SupportsBfloat16FastPerformance() {
#ifndef PADDLE_WITH_DNNL
  return false;
#else
  if (phi::backends::cpu::MayIUse(phi::backends::cpu::cpu_isa_t::avx512_bf16))
    return true;
  else
    return false;
#endif
}

bool SupportsInt8() {
#ifndef PADDLE_WITH_DNNL
  return false;
#else
  return (phi::backends::cpu::MayIUse(phi::backends::cpu::cpu_isa_t::avx2) ||
          phi::backends::cpu::MayIUse(phi::backends::cpu::cpu_isa_t::avx512f));
#endif
}

bool SupportsAvx512F() {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::cpu_isa_t::avx512f);
}

bool SupportsVNNI() {
#ifndef PADDLE_WITH_DNNL
  return false;
#else
  return phi::backends::cpu::MayIUse(
      phi::backends::cpu::cpu_isa_t::avx512_core_vnni);
#endif
}

bool IsCompiledWithBrpc() {
#ifndef PADDLE_WITH_DISTRIBUTE
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithDIST() {
#ifdef PADDLE_WITH_DISTRIBUTE
  return true;
#else
  return false;
#endif
}

struct iinfo {
  int64_t min, max;
  int bits;
  std::string dtype;

  explicit iinfo(const framework::proto::VarType::Type &type) {
    switch (type) {
      case framework::proto::VarType::INT16:
        min = std::numeric_limits<int16_t>::min();
        max = std::numeric_limits<int16_t>::max();
        bits = 16;
        dtype = "int16";
        break;
      case framework::proto::VarType::INT32:
        min = std::numeric_limits<int32_t>::min();
        max = std::numeric_limits<int32_t>::max();
        bits = 32;
        dtype = "int32";
        break;
      case framework::proto::VarType::INT64:
        min = std::numeric_limits<int64_t>::min();
        max = std::numeric_limits<int64_t>::max();
        bits = 64;
        dtype = "int64";
        break;
      case framework::proto::VarType::INT8:
        min = std::numeric_limits<int8_t>::min();  // NOLINT
        max = std::numeric_limits<int8_t>::max();
        bits = 8;
        dtype = "int8";
        break;
      case framework::proto::VarType::UINT8:
        min = std::numeric_limits<uint8_t>::min();
        max = std::numeric_limits<uint8_t>::max();
        bits = 8;
        dtype = "uint8";
        break;
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "the argument of paddle.iinfo can only be paddle.int8, "
            "paddle.int16, paddle.int32, paddle.int64, or paddle.uint8"));
        break;
    }
  }
};

struct finfo {
  int64_t bits;
  double eps;
  double min;  // lowest()
  double max;
  double tiny;
  double smallest_normal;  // min()
  double resolution;
  std::string dtype;

  explicit finfo(const framework::proto::VarType::Type &type) {
    switch (type) {
      case framework::proto::VarType::FP16:
        eps = std::numeric_limits<phi::dtype::float16>::epsilon();
        min = std::numeric_limits<phi::dtype::float16>::lowest();
        max = std::numeric_limits<phi::dtype::float16>::max();
        smallest_normal = std::numeric_limits<phi::dtype::float16>::min();
        tiny = smallest_normal;
        resolution =
            std::pow(10, -std::numeric_limits<phi::dtype::float16>::digits10);
        bits = 16;
        dtype = "float16";
        break;
      case framework::proto::VarType::FP32:
      case framework::proto::VarType::COMPLEX64:
        eps = std::numeric_limits<float>::epsilon();
        min = std::numeric_limits<float>::lowest();
        max = std::numeric_limits<float>::max();
        smallest_normal = std::numeric_limits<float>::min();
        tiny = smallest_normal;
        resolution = std::pow(10, -std::numeric_limits<float>::digits10);
        bits = 32;
        dtype = "float32";
        break;
      case framework::proto::VarType::FP64:
      case framework::proto::VarType::COMPLEX128:
        eps = std::numeric_limits<double>::epsilon();
        min = std::numeric_limits<double>::lowest();
        max = std::numeric_limits<double>::max();
        smallest_normal = std::numeric_limits<double>::min();
        tiny = smallest_normal;
        resolution = std::pow(10, -std::numeric_limits<double>::digits10);
        bits = 64;
        dtype = "float64";
        break;
      case framework::proto::VarType::BF16:
        eps = std::numeric_limits<phi::dtype::bfloat16>::epsilon();
        min = std::numeric_limits<phi::dtype::bfloat16>::lowest();
        max = std::numeric_limits<phi::dtype::bfloat16>::max();
        smallest_normal = std::numeric_limits<phi::dtype::bfloat16>::min();
        tiny = smallest_normal;
        resolution =
            std::pow(10, -std::numeric_limits<phi::dtype::bfloat16>::digits10);
        bits = 16;
        dtype = "bfloat16";
        break;
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "the argument of paddle.finfo can only be paddle.float32, "
            "paddle.float64, paddle.float16, paddle.bfloat16"
            "paddle.complex64, or paddle.complex128"));
        break;
    }
  }
};

static PyObject *GetPythonAttribute(PyObject *obj, const char *attr_name) {
  // NOTE(zjl): PyObject_GetAttrString would return nullptr when attr_name
  // is not inside obj, but it would also set the error flag of Python.
  // If the error flag is set in C++, C++ code would not raise Exception,
  // but Python would raise Exception once C++ call ends.
  // To avoid unexpected Exception raised in Python, we check whether
  // attribute exists before calling PyObject_GetAttrString.
  //
  // Caution: PyObject_GetAttrString would increase reference count of PyObject.
  // Developer should call Py_DECREF manually after the attribute is not used.
  if (PyObject_HasAttrString(obj, attr_name)) {
    return PyObject_GetAttrString(obj, attr_name);
  } else {
    return nullptr;
  }
}

using PyNameVarBaseMap = std::unordered_map<std::string, py::handle>;

static std::vector<std::shared_ptr<imperative::VarBase>> GetVarBaseList(
    const PyNameVarBaseMap &state_dict) {
  std::vector<std::shared_ptr<imperative::VarBase>> vec_res;
  vec_res.reserve(state_dict.size());

  for (auto &para : state_dict) {
    PyObject *py_obj = para.second.ptr();
    if (!py_obj || py_obj == Py_None) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The parameter [%s] to save is None", para.first));
    }
    vec_res.emplace_back(
        PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_obj));
  }

  return vec_res;
}

static std::vector<std::string> inline GetNameList(
    const py::handle &py_handle) {
  std::vector<std::string> vec_res;

  PyObject *py_obj = py_handle.ptr();  // get underlying PyObject
  // Python None is not nullptr in C++!
  if (!py_obj || py_obj == Py_None) {
    PADDLE_THROW(
        common::errors::InvalidArgument("The parameter list to save is None"));
  }

  if (PyList_Check(py_obj)) {
    size_t len = PyList_GET_SIZE(py_obj);

    vec_res.reserve(len);

    const char *kNameField = "name";

    for (size_t i = 0; i < len; ++i) {
      PyObject *py_name =
          PyObject_GetAttrString(PyList_GET_ITEM(py_obj, i), kNameField);
      PADDLE_ENFORCE_NOT_NULL(py_name,
                              common::errors::InvalidArgument(
                                  "The name of parameter to save is None"));
      vec_res.emplace_back(PyObjectCast<std::string>(py_name));
      Py_DECREF(py_name);
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The parameters to save is not a list"));
  }
  return vec_res;
}

static void inline CreateVariableIfNotExist(
    const py::handle &py_handle,
    const framework::Scope &scope,
    const framework::Executor *exe = nullptr) {
  std::vector<std::string> vec_res;

  PyObject *py_obj = py_handle.ptr();  // get underlying PyObject
  // Python None is not nullptr in C++!
  if (!py_obj || py_obj == Py_None) {
    PADDLE_THROW(
        common::errors::InvalidArgument("The parameter list to set is None"));
  }

  if (PyList_Check(py_obj)) {
    size_t len = PyList_GET_SIZE(py_obj);

    vec_res.reserve(len);

    const char *kNameField = "name";
    const char *kVarDescField = "desc";

    for (size_t i = 0; i < len; ++i) {
      PyObject *py_name =
          PyObject_GetAttrString(PyList_GET_ITEM(py_obj, i), kNameField);
      PADDLE_ENFORCE_NOT_NULL(py_name,
                              common::errors::InvalidArgument(
                                  "The name of parameter to set is None"));
      auto para_name = PyObjectCast<std::string>(py_name);
      Py_DECREF(py_name);

      auto var = scope.FindVar(para_name);
      if (var == nullptr) {
        PADDLE_ENFORCE_NOT_NULL(exe,
                                common::errors::InvalidArgument(
                                    "Parameter not Initialized, "
                                    "Please set argument [executor] not None "
                                    "or run startup program first"));
        PyObject *py_var_desc =
            PyObject_GetAttrString(PyList_GET_ITEM(py_obj, i), kVarDescField);
        PADDLE_ENFORCE_NOT_NULL(
            py_var_desc,
            common::errors::InvalidArgument(
                "The var_desc of parameter to set is None"));
        auto var_desc = PyObjectCast<framework::VarDesc>(py_var_desc);
        Py_DECREF(py_var_desc);
        var = const_cast<framework::Scope *>(&scope)->Var(para_name);
        auto *tensor_temp = var->GetMutable<phi::DenseTensor>();
        tensor_temp->Resize(common::make_ddim(var_desc.GetShape()));
        tensor_temp->mutable_data(
            exe->GetPlace(), phi::TransToPhiDataType(var_desc.GetDataType()));
      }
    }
  } else {
    PADDLE_THROW(
        common::errors::InvalidArgument("The parameters to set is not a list"));
  }

  return;
}

static void AssertStaticGraphAndDygraphGradMakerNoDiff() {
  std::set<std::string> ops;
  for (auto &pair : framework::OpInfoMap::Instance().map()) {
    bool has_static_grad_maker = (pair.second.grad_op_maker_ != nullptr);
    bool has_dygraph_grad_maker =
        (pair.second.dygraph_grad_op_maker_ != nullptr);
    if (has_static_grad_maker ^ has_dygraph_grad_maker) {
      bool has_kernel =
          (framework::OperatorWithKernel::AllOpKernels().count(pair.first) > 0);
      if (has_kernel) {
        ops.insert(pair.first);
      } else {
        VLOG(5) << pair.first << " has no kernels, skip";
      }
    }
  }
  PADDLE_ENFORCE_EQ(ops.empty(),
                    true,
                    common::errors::Unimplemented(
                        "OperatorWithKernel [%s] have only static graph grad "
                        "maker or have only dygraph grad maker, which is not "
                        "allowed",
                        string::join_strings(ops, ',')));
}

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
static int GetNCCLVersion() {
#if NCCL_VERSION_CODE >= 2304
  int ver;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGetVersion(&ver));
  return ver;
#else
  PADDLE_THROW(common::errors::External(
      "Cannot get NCCL version successfully when nccl version < 2.3.4"));
#endif
}
#endif

// NOTE: Use to manage the context of pylayer op constructing block
class PyLayerBlockContextManager {
 public:
  explicit PyLayerBlockContextManager(pir::Block *block) {
    dialect::ApiBuilder::Instance().PushInsertionPoint();
    dialect::ApiBuilder::Instance().SetInsertionPointToBlockEnd(block);
  }

  ~PyLayerBlockContextManager() {
    dialect::ApiBuilder::Instance().LoadInsertionPoint();
  }

  PyLayerBlockContextManager(const PyLayerBlockContextManager &) = delete;
  PyLayerBlockContextManager &operator=(const PyLayerBlockContextManager &) =
      delete;

 private:
  // disable default constructor
  PyLayerBlockContextManager() = default;
};

static std::vector<std::vector<pir::Value>> GenerateBackwardBlockForPyLayerOp(
    pir::Operation *op,
    const std::vector<std::vector<pir::Value>> &inputs_,
    const std::vector<std::vector<pir::Value>> &outputs,
    const std::vector<std::vector<pir::Value>> &out_grads,
    const std::vector<std::vector<bool>> &stop_gradients) {
  PADDLE_ENFORCE(
      op->isa<paddle::dialect::PyLayerOp>(),
      common::errors::InvalidArgument(
          "GenerateBackwardBlockForPyLayerOp only support PyLayerOp"));

  // 1. construct pylayer grad op
  VLOG(6) << "Prepare Outputs for pylayer_grad";
  std::vector<pir::Type> output_types;
  // NOTE: the last input of pylayer op is create_stack when called
  // save_for_backward, whose stop_gradient is always True
  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i][0].type().isa<pir::InletType>()) break;
    output_types.push_back(inputs_[i][0].type());
  }

  VLOG(6) << "Prepare Inputs for pylayer_grad";
  std::vector<pir::Value> output_grads;
  for (size_t i = 0; i < out_grads.size(); ++i) {
    output_grads.push_back(out_grads[i][0]);
  }

  std::vector<pir::Value> pylayer_grad_inputs(output_types.size());
  auto pylayer_grad = dialect::ApiBuilder::Instance()
                          .GetBuilder()
                          ->Build<paddle::dialect::PyLayerOp>(
                              output_grads, std::move(output_types), -1);

  VLOG(6) << "Construct pylayer_grad finished";

  // 2.1 Get registered backward function from
  // `PythonCallableRegistrar::python_callable_registry_`.
  int backward_function_id =
      op->attributes()
          .at(paddle::dialect::PyLayerOp::kBackwardFunctionIdAttrName)
          .dyn_cast<pir::Int32Attribute>()
          .data();
  PADDLE_ENFORCE_GE(
      backward_function_id,
      0,
      common::errors::InvalidArgument("The backward function id of pylayer op "
                                      "should be non-negative, but got %d",
                                      backward_function_id));
  VLOG(6) << "pylayer op unique_id is " << op->id();
  VLOG(6) << "pylayer op backward_function_id is " << backward_function_id;
  auto py_callable = paddle::pybind::PythonCallableRegistrar::GetInstance().Get(
      static_cast<uint64_t>(backward_function_id));

  // 2.2 Get TuplePushOp from forward block if exists
  auto pylayer_op = op->dyn_cast<paddle::dialect::PyLayerOp>();
  std::vector<pir::Operation *> tuple_push_op_list;
  for (auto &op : pylayer_op.forward_block()) {
    if (op.isa<pir::TuplePushOp>()) {
      tuple_push_op_list.push_back(&op);
    }
  }
  PADDLE_ENFORCE_LE(tuple_push_op_list.size(),
                    1,
                    common::errors::InvalidArgument(
                        "The number of tuple_push op in pylayer forward block "
                        "is either unique or does not exist."));

  {
    // enter block of pylayer_grad
    PyLayerBlockContextManager pylayer_block_context_manager(
        &(pylayer_grad.forward_block()));

    // create tuple_pop op if needed
    if (tuple_push_op_list.size() > 0) {
      VLOG(6) << "Start creating tuple_pop op in the front of backward block "
                 "of pylayer.";
      auto tuple_push_op = tuple_push_op_list[0]->dyn_cast<pir::TuplePushOp>();
      dialect::ApiBuilder::Instance().GetBuilder()->Build<pir::TuplePopOp>(
          tuple_push_op.outlet());
      VLOG(6) << "Finish creating tuple_pop op.";
    }

    VLOG(6) << "call pylayer op backward function";
    PirCallPythonFunc(py_callable, output_grads, &pylayer_grad_inputs);

    // append yield op for outputs value
    dialect::ApiBuilder::Instance().GetBuilder()->Build<pir::YieldOp>(
        pylayer_grad_inputs);
    // exit block of pylayer_grad
  }
  VLOG(6) << "Construct pylayer backward block finished";

  // 3. Update pylayer_grad op's attributes of outputs
  pylayer_grad.UpdateOutput();
  VLOG(6) << "Update pylayer_grad op finished";

  std::vector<std::vector<pir::Value>> res{inputs_.size()};
  for (size_t i = 0; i < res.size(); ++i) {
    res[i].resize(1);
    res[i][0] = !stop_gradients[i][0] ? pylayer_grad->result(i) : pir::Value();
  }
  return res;
}

void BindVjp(pybind11::module *m) {
  m->def(
      "call_vjp",
      [](pir::Operation &fwd_op,
         const std::vector<std::vector<pir::Value>> &inputs,
         const std::vector<std::vector<pir::Value>> &outputs,
         const std::vector<std::vector<pir::Value>> &out_grads,
         const std::vector<std::vector<bool>> &stop_gradients) {
        // NOTE(dev): Prim decomposed rules will call paddle::dialect::xx
        // api, which has amp strategy. But Prim already process cast operation
        // and we need to disable amp strategy here.
        paddle::imperative::AutoCastGuard guard(
            egr::Controller::Instance().GetCurrentAmpAttrs(),
            paddle::imperative::AmpLevel::O0);

        py::list res;
        std::vector<std::vector<pir::Value>> vjp_res;

        if (fwd_op.isa<paddle::dialect::PyLayerOp>()) {
          // NOTE(MarioLulab): In PIR mode, even though the `PyLayer` op does
          // not have a vjp interface, we still need to generate the backward
          // block based on its registered backward function.
          vjp_res = GenerateBackwardBlockForPyLayerOp(
              &fwd_op, inputs, outputs, out_grads, stop_gradients);
        } else {
          paddle::dialect::VjpInterface vjp_interface =
              fwd_op.dyn_cast<paddle::dialect::VjpInterface>();
          PADDLE_ENFORCE(vjp_interface,
                         common::errors::InvalidArgument(
                             "The vjp function is not registered in %s op ",
                             fwd_op.name()));
          vjp_res = vjp_interface.Vjp(
              &fwd_op, inputs, outputs, out_grads, stop_gradients);
        }

        PADDLE_ENFORCE_EQ(
            stop_gradients.size(),
            vjp_res.size(),
            common::errors::InvalidArgument(
                "The size of  %s stop_gradients should be the same as vjp_res "
                "size."
                "But the size of stop_gradients: %d, vjp_res size: %d",
                fwd_op.name(),
                stop_gradients.size(),
                vjp_res.size()));

        for (size_t i = 0; i < vjp_res.size(); ++i) {
          PADDLE_ENFORCE_EQ(stop_gradients[i].size(),
                            vjp_res[i].size(),
                            common::errors::InvalidArgument(
                                "The size of stop_gradients[%d] should be the "
                                "same as vjp_res[%d] "
                                "size."
                                "But the size of stop_gradients[%d]: %d, "
                                "vjp_res[%d] size: %d",
                                i,
                                i,
                                i,
                                stop_gradients[i].size(),
                                i,
                                vjp_res[i].size()));
          py::list sub_res;
          for (size_t j = 0; j < vjp_res[i].size(); ++j) {
            if (!vjp_res[i][j]) {
              sub_res.append(nullptr);
            } else {
              // The grad_type must equal to forward type.
              sub_res.append(vjp_res[i][j]);
            }
          }
          res.append(sub_res);
        }

        paddle::dialect::OpYamlInfoInterface yaml_interface =
            fwd_op.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
        if (yaml_interface) {
          auto inputs_grad_info = std::get<0>(yaml_interface.GetOpInfo());
          PADDLE_ENFORCE_EQ(inputs.size(),
                            inputs_grad_info.size(),
                            common::errors::InvalidArgument(
                                "The size of %s inputs should be the "
                                "same as inputs_grad_info size.",
                                fwd_op.name()));
          size_t grad_index = 0;
          for (size_t idx = 0; idx < inputs.size(); ++idx) {
            if (!inputs_grad_info[idx].with_grad_semantic) continue;
            PADDLE_ENFORCE_EQ(inputs[idx].size(),
                              vjp_res[grad_index].size(),
                              common::errors::InvalidArgument(
                                  "The size of inputs[%d] should be the "
                                  "same as vjp_res[%d] size.",
                                  idx,
                                  grad_index));
            for (size_t j = 0; j < inputs[idx].size(); ++j) {
              if (vjp_res[grad_index][j]) {
                // The grad_type must equal to forward type.
                if (auto fwd_type =
                        inputs[idx][j]
                            .type()
                            .dyn_cast<dialect::DistTypeInterface>()) {
                  if (auto bwd_type =
                          vjp_res[grad_index][j]
                              .type()
                              .dyn_cast<dialect::DistTypeInterface>()) {
                    auto fwd_attr = fwd_type.tensor_dist_attr();
                    auto bwd_attr = bwd_type.tensor_dist_attr();
                    if (fwd_attr.process_mesh_attr() ==
                            bwd_attr.process_mesh_attr() &&
                        fwd_attr.dims_mapping() == bwd_attr.dims_mapping()) {
                      continue;
                    }
                  }
                }
                vjp_res[grad_index][j].set_type(inputs[idx][j].type());
              }
            }
            ++grad_index;
          }
        }
        return res;
      });

  m->def("has_vjp", [](pir::Operation &fwd_op) {
    pir::IrContext *ctx = pir::IrContext::Instance();
    pir::OpInfo fwd_op_info = ctx->GetRegisteredOpInfo(fwd_op.name());
    auto vjp_interface_impl =
        fwd_op_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();
    if (vjp_interface_impl == nullptr) return false;
    return true;
  });

  m->def(
      "has_custom_vjp",
      [](pir::Operation &op) -> py::bool_ {
        return op.info().HasTrait<paddle::dialect::CustomVjpTrait>();
      },
      R"DOC(
           Return whether an op has custom vjp rules.

           Args:
               op (pir::Operation): op to be checked

           Returns:
               out (bool): True means that the op has custom vjp rules, False means it does not.
           )DOC");
  m->def(
      "is_forward_only",
      [](pir::Operation &op) -> py::bool_ {
        return op.info().HasTrait<paddle::dialect::ForwardOnlyTrait>();
      },
      R"DOC(
           Return whether an op is forward only op.

           Args:
               op (pir::Operation): op to be checked

           Returns:
               out (bool): True means that the op is forward only op, False means it does not.
           )DOC");
}

void BindDecompRule(pybind11::module *m) {
  m->def("sinking_decomp",
         [](pir::Program *program,
            std::vector<pir::Value> &src_vars,
            std::set<std::string> &blacklist,
            std::set<std::string> &whitelist,
            int start_index,
            int end_index) {
           VLOG(4) << "[Prim] Bind Decomp sinking_decomp begin.";
           py::list res;
           DecompProgram decomp_object(
               program, src_vars, blacklist, whitelist, start_index, end_index);
           decomp_object.decomp_program();
           std::vector<pir::Value> tar_vars = decomp_object.get_dst_vars();
           for (size_t i = 0; i < tar_vars.size(); ++i) {
             if (!tar_vars[i]) {
               res.append(nullptr);
             } else {
               res.append(tar_vars[i]);
             }
           }
           VLOG(4) << "[Prim] Bind Decomp sinking_decomp end.";
           return res;
         });

  m->def("call_decomp_rule", [](pir::Operation &fwd_op) {
    py::list res;
    std::vector<std::vector<pir::Value>> decomp_res = call_decomp_rule(&fwd_op);
    for (size_t i = 0; i < decomp_res.size(); ++i) {
      py::list sub_res;
      for (size_t j = 0; j < decomp_res[i].size(); ++j) {
        if (!decomp_res[i][j]) {
          sub_res.append(nullptr);
        } else {
          sub_res.append(decomp_res[i][j]);
        }
      }
      res.append(sub_res);
    }
    return res;
  });

  m->def("has_decomp_rule", [](pir::Operation &fwd_op) {
    return paddle::has_decomp_rule(fwd_op);
  });
}

void BindDecompVjp(pybind11::module *m) {
  m->def("call_decomp_vjp", [](pir::Operation &vjp_op) {
    py::list res;
    std::vector<std::vector<pir::Value>> decomp_res = call_decomp_vjp(&vjp_op);
    for (size_t i = 0; i < decomp_res.size(); ++i) {
      py::list sub_res;
      for (size_t j = 0; j < decomp_res[i].size(); ++j) {
        sub_res.append(decomp_res[i][j]);
      }
      res.append(sub_res);
    }
    return res;
  });

  m->def("has_decomp_vjp",
         [](pir::Operation &vjp_op) { return paddle::has_decomp_vjp(vjp_op); });
}

PYBIND11_MODULE(libpaddle, m) {
  BindImperative(&m);
  BindEager(&m);
  BindEagerStringTensor(&m);
  BindCudaStream(&m);
  BindXpuStream(&m);
  BindJit(&m);
  BindSot(&m);
  BindCustomDevicePy(&m);
  BindEagerUtils(m.ptr());

  // Not used, just make sure cpu_info.cc is linked.
  phi::backends::cpu::CpuTotalPhysicalMemory();

  paddle::memory::allocation::UseAllocatorStrategyGFlag();

  AssertStaticGraphAndDygraphGradMakerNoDiff();

  m.doc() = "C++ core of PaddlePaddle";

  // using framework in this function. Since it is inside a function, it will
  // not cause namespace pollution.
  using namespace paddle::framework;  // NOLINT

  BindException(&m);

  py::class_<iinfo>(m, "iinfo")
      .def(py::init<const framework::proto::VarType::Type &>())
      .def_readonly("min", &iinfo::min)
      .def_readonly("max", &iinfo::max)
      .def_readonly("bits", &iinfo::bits)
      .def_readonly("dtype", &iinfo::dtype)
      .def("__repr__", [](const iinfo &a) {
        std::ostringstream oss;
        oss << "paddle.iinfo(min=" << a.min;
        oss << ", max=" << a.max;
        oss << ", bits=" << a.bits;
        oss << ", dtype=" << a.dtype << ")";
        return oss.str();
      });

  py::class_<finfo>(m, "finfo")
      .def(py::init<const framework::proto::VarType::Type &>())
      .def_readonly("min", &finfo::min)
      .def_readonly("max", &finfo::max)
      .def_readonly("bits", &finfo::bits)
      .def_readonly("eps", &finfo::eps)
      .def_readonly("resolution", &finfo::resolution)
      .def_readonly("smallest_normal", &finfo::smallest_normal)
      .def_readonly("tiny", &finfo::tiny)
      .def_readonly("dtype", &finfo::dtype)
      .def("__repr__", [](const finfo &a) {
        std::ostringstream oss;
        oss << "paddle.finfo(min=" << a.min;
        oss << ", max=" << a.max;
        oss << ", eps=" << a.eps;
        oss << ", resolution=" << a.resolution;
        oss << ", smallest_normal=" << a.smallest_normal;
        oss << ", tiny=" << a.tiny;
        oss << ", bits=" << a.bits;
        oss << ", dtype=" << a.dtype << ")";
        return oss.str();
      });

  m.def("__set_bwd_prim_enabled",
        &paddle::prim::PrimCommonUtils::SetBwdPrimEnabled);
  m.def("_is_bwd_prim_enabled",
        &paddle::prim::PrimCommonUtils::IsBwdPrimEnabled);
  m.def("__set_fwd_prim_enabled",
        &paddle::prim::PrimCommonUtils::SetFwdPrimEnabled);
  m.def("_is_fwd_prim_enabled",
        &paddle::prim::PrimCommonUtils::IsFwdPrimEnabled);
  m.def("__set_all_prim_enabled",
        &paddle::prim::PrimCommonUtils::SetAllPrimEnabled);
  m.def("_is_eager_prim_enabled",
        &paddle::prim::PrimCommonUtils::IsEagerPrimEnabled);
  m.def("__set_eager_prim_enabled",
        &paddle::prim::PrimCommonUtils::SetEagerPrimEnabled);
  m.def("_set_prim_target_grad_name",
        &paddle::prim::PrimCommonUtils::SetTargetGradName);
  m.def("set_num_threads", &platform::SetNumThreads);

  m.def("need_type_promotion_old_ir",
        [](const std::string &op_name,
           framework::proto::VarType::Type type_x,
           framework::proto::VarType::Type type_y) {
          return phi::NeedTypePromotionOldIr(op_name,
                                             phi::TransToPhiDataType(type_x),
                                             phi::TransToPhiDataType(type_y));
        });
  m.def("get_promote_dtype_old_ir",
        [](const std::string &op_name,
           framework::proto::VarType::Type type_x,
           framework::proto::VarType::Type type_y) {
          return framework::TransToProtoVarType(
              phi::GetPromoteDtypeOldIr(op_name,
                                        phi::TransToPhiDataType(type_x),
                                        phi::TransToPhiDataType(type_y)));
        });
  m.def("is_common_dtype_for_scalar",
        [](framework::proto::VarType::Type type_x,
           framework::proto::VarType::Type type_y) {
          return phi::is_common_dtype_for_scalar(
              phi::TransToPhiDataType(type_x), phi::TransToPhiDataType(type_y));
        });
  m.def("disable_signal_handler", &DisableSignalHandler);

  m.def("clear_gradients",
        [](std::vector<std::shared_ptr<imperative::VarBase>> param_list,
           bool set_to_zero) {
          for (auto const &param : param_list) {
            param->ClearGradient(set_to_zero);
          }
        });

  class NodePostHookRemoveHelper {
   public:
    NodePostHookRemoveHelper(std::shared_ptr<egr::GradNodeBase> node,
                             int64_t hook_id)
        : node_(node), hook_id_(hook_id) {}
    ~NodePostHookRemoveHelper() = default;
    bool remove() { return node_->RemoveNodePostHook(hook_id_); }

   private:
    std::shared_ptr<egr::GradNodeBase> node_;
    int64_t hook_id_;
  };

  py::class_<NodePostHookRemoveHelper,
             std::shared_ptr<NodePostHookRemoveHelper>>(
      m, "NodePostHookRemoveHelper")
      .def("remove", &NodePostHookRemoveHelper::remove);

  py::class_<egr::GradNodeBase, std::shared_ptr<egr::GradNodeBase>>(
      m, "GradNodeBase")
      .def("name",
           [](const std::shared_ptr<egr::GradNodeBase> &self) {
             return self->name();
           })
      .def_property_readonly(
          "next_functions",
          [](const std::shared_ptr<egr::GradNodeBase> &self) {
            return self->NextFunctions();
          })
      .def("node_ptr", &egr::GradNodeBase::GetPtr)
      .def("input_meta",
           [](const std::shared_ptr<egr::GradNodeBase> &self) {
             return self->InputMeta();
           })
      .def("output_meta",
           [](const std::shared_ptr<egr::GradNodeBase> &self) {
             return self->OutputMeta();
           })
      .def("_register_post_hook",
           [](const std::shared_ptr<egr::GradNodeBase> &self, py::object hook) {
             if (std::dynamic_pointer_cast<egr::GradNodeAccumulation>(self)) {
               PADDLE_THROW(common::errors::InvalidArgument(
                   "Could not register hook for GradNodeAccumulation."));
             }
             int64_t hook_id = self->RegisterNodePostHook(
                 std::make_shared<NodePostHook>(hook));
             return std::make_shared<NodePostHookRemoveHelper>(self, hook_id);
           });

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  m.def("cudnn_version", &platform::DnnVersion);
  m.def("gpu_memory_available", []() {
    size_t available = 0;
    size_t total = 0;
    paddle::platform::GpuMemoryUsage(&available, &total);
    return available;
  });
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  m.def("nccl_version", &GetNCCLVersion);
#endif

  m.def("is_cuda_graph_capturing", &platform::IsCUDAGraphCapturing);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  py::class_<phi::backends::gpu::CUDAGraph>(m, "CUDAGraph")
      .def_static("begin_capture",
                  [](phi::GPUPlace place, int mode) {
                    platform::BeginCUDAGraphCapture(
                        place, static_cast<paddle::gpuStreamCaptureMode>(mode));
                  })
      .def_static("end_capture", &platform::EndCUDAGraphCapture)
      .def_static("gen_new_memory_pool_id",
                  &phi::backends::gpu::CUDAGraph::UniqueMemoryPoolID)
      .def("replay", &phi::backends::gpu::CUDAGraph::Replay)
      .def("reset", &phi::backends::gpu::CUDAGraph::Reset)
      .def("print_to_dot_files",
           &phi::backends::gpu::CUDAGraph::PrintToDotFiles);
#endif

  m.def("wait_device", [](const phi::Place &place) {
    phi::DeviceContextPool::Instance().Get(place)->Wait();
  });

  m.def("from_dlpack", [](py::object data) {
    DLManagedTensor *dlMTensor = reinterpret_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(data.ptr(), "dltensor"));

    PADDLE_ENFORCE_NOT_NULL(
        dlMTensor,
        common::errors::InvalidArgument(
            "from_dlpack received an invalid capsule. "
            "Note that DLTensor capsules can be consumed only once, "
            "so you might have already constructed a tensor from it once."));

    // NOTE: Might meet bugged numpy version, see:
    // https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_new.cpp#L1636-L1638
    auto ptensor = paddle::framework::TensorFromDLPack(dlMTensor);

    PyCapsule_SetName(data.ptr(), "used_dltensor");
    return ptensor;
  });

  m.def("tensor_from_cuda_array_interface", [](py::object obj) {
    // We use CUDA Array Interface (Version 2) protocol:
    // https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
    py::object cuda_array_interface = obj.attr("__cuda_array_interface__");
    PADDLE_ENFORCE_EQ(py::isinstance<py::dict>(cuda_array_interface),
                      true,
                      common::errors::InvalidArgument(
                          "`__cuda_array_interface` must be a dict"));
    py::dict cuda_dict = cuda_array_interface.cast<py::dict>();

    // Extract the `obj.__cuda_array_interface__['shape']` attribute
    PADDLE_ENFORCE_EQ(
        cuda_dict.contains("shape"),
        true,
        common::errors::InvalidArgument(
            "The 'shape' key is missing in the __cuda_array_interface__  "
            "dict."));
    py::object shape_obj = cuda_dict["shape"];
    PADDLE_ENFORCE_EQ(
        py::isinstance<py::tuple>(shape_obj) ||
            py::isinstance<py::list>(shape_obj),
        true,
        common::errors::InvalidArgument("Shape must be a tuple or list"));
    std::vector<int64_t> shapes;
    shapes = shape_obj.cast<std::vector<int64_t>>();
    phi::IntArray shapeIntArray = phi::IntArray(shapes);

    // Extract the `obj.__cuda_array_interface__['typestr'] attribute
    PADDLE_ENFORCE_EQ(
        cuda_dict.contains("typestr"),
        true,
        common::errors::InvalidArgument(
            "The 'typestr' key is missing in the __cuda_array_interface__  "
            "dict."));
    py::object typestr_obj = cuda_dict["typestr"];
    std::string typestr = typestr_obj.cast<std::string>();
    phi::DataType dtype = paddle::framework::ConvertToPDDataType(typestr);

    // Extract the `obj.__cuda_array_interface__['data']` attribute
    PADDLE_ENFORCE_EQ(
        cuda_dict.contains("data"),
        true,
        common::errors::InvalidArgument(
            "The 'data' key is missing in the __cuda_array_interface__  "
            "dict."));
    py::object data_obj = cuda_dict["data"];
    py::tuple data_tuple = data_obj.cast<py::tuple>();

    // Data tuple(ptr_as_int, read_only_flag).
    // The ptr_as_int stands for data pointer but in Python it is a integer.
    // It need to be converted to a large enough integral type first
    // and then convert to void*
    void *data_ptr = reinterpret_cast<void *>(data_tuple[0].cast<intptr_t>());
    PADDLE_ENFORCE_NE(
        data_tuple[1].cast<bool>(),
        true,
        common::errors::InvalidArgument("Read-only array is not supported"));

    // Extract the `obj.__cuda_array_interface__['strides']` attribute
    phi::IntArray stridesIntArray;
    if (cuda_dict.contains("strides") && !cuda_dict["strides"].is_none()) {
      std::vector<int64_t> strides_vec =
          cuda_dict["strides"].cast<std::vector<int64_t>>();

      // __cuda_array_interface__ strides uses bytes
      size_t element_size = phi::SizeOf(dtype);
      for (auto &stride : strides_vec) {
        PADDLE_ENFORCE_NE(
            stride % element_size,
            0,
            common::errors::InvalidArgument(
                "strides must be a multiple of the element size."));
        stride /= element_size;
      }
      stridesIntArray = phi::IntArray(strides_vec);
    } else {
      DDim ddim_strides =
          phi::DenseTensorMeta::calc_strides(common::make_ddim(shapes));
      int rank = ddim_strides.size();
      const int64_t *ddim_data = ddim_strides.Get();
      std::vector<int64_t> strides_vec(ddim_data, ddim_data + rank);
      stridesIntArray = phi::IntArray(strides_vec);
    }
    return paddle::from_blob(data_ptr,
                             shapeIntArray,
                             stridesIntArray,
                             dtype,
                             phi::DataLayout::NCHW,
                             phi::Place(),
                             [obj](void *data) {
                               py::gil_scoped_acquire gil;
                               obj.dec_ref();
                             });
  });

  m.def("_create_loaded_parameter",
        [](const py::handle &vec_var_list,
           const Scope &scope,
           const Executor *executor) {
          CreateVariableIfNotExist(vec_var_list, scope, executor);
        });

  m.def("save_op_version_info", [](framework::ProgramDesc &desc) {
    framework::compatible::pb::OpVersionMap pb_vmap{desc.OpVersionMap()};
    framework::compatible::SaveOpVersions(
        framework::compatible::OpVersionRegistrar::GetInstance()
            .GetVersionMap(),
        &pb_vmap);
  });

  m.def("set_printoptions", [](const py::kwargs &kwargs) {
    auto &print_opt = framework::PrintOptions::Instance();
    if (kwargs.contains("precision")) {
      print_opt.precision = kwargs["precision"].cast<int>();
    }
    if (kwargs.contains("threshold")) {
      print_opt.threshold = kwargs["threshold"].cast<int>();
    }
    if (kwargs.contains("edgeitems")) {
      print_opt.edgeitems = kwargs["edgeitems"].cast<int>();
    }
    if (kwargs.contains("linewidth")) {
      print_opt.linewidth = kwargs["linewidth"].cast<int>();
    }
    if (kwargs.contains("sci_mode")) {
      print_opt.sci_mode = kwargs["sci_mode"].cast<bool>();
    }

    VLOG(4) << "Set printoptions: precision=" << print_opt.precision
            << ", threshold=" << print_opt.threshold
            << ", edgeitems=" << print_opt.edgeitems
            << ", linewidth=" << print_opt.linewidth
            << ", sci_mode=" << print_opt.sci_mode;
  });

  m.def(
      "broadcast_shape",
      [](const std::vector<int64_t> &x_dim, const std::vector<int64_t> &y_dim) {
        return common::vectorize(phi::funcs::BroadcastTwoDims(
            common::make_ddim(x_dim), common::make_ddim(y_dim), -1));
      });

  m.def(
      "_append_python_callable_object_and_return_id",
      [](py::object py_obj) -> size_t {
        return paddle::operators::AppendPythonCallableObjectAndReturnId(py_obj);
      });

  m.def("_get_use_default_grad_op_desc_maker_ops",
        [] { return OpInfoMap::Instance().GetUseDefaultGradOpDescMakerOps(); });

  m.def(
      "_get_all_register_op_kernels",
      [](const std::string &lib) {
        std::unordered_map<std::string, std::vector<std::string>>
            all_kernels_info;
        if (lib == "fluid" || lib == "all") {
          auto &all_kernels =
              paddle::framework::OperatorWithKernel::AllOpKernels();

          for (auto &kernel_pair : all_kernels) {
            auto op_type = kernel_pair.first;
            std::vector<std::string> kernel_types;
            for (auto &info_pair : kernel_pair.second) {
              paddle::framework::OpKernelType kernel_type = info_pair.first;
              kernel_types.emplace_back(
                  paddle::framework::KernelTypeToString(kernel_type));
            }
            all_kernels_info.emplace(op_type, kernel_types);
          }
        }
        if (lib == "phi" || lib == "all") {
          auto phi_kernels = phi::KernelFactory::Instance().kernels();
          for (auto &kernel_pair : phi_kernels) {
            auto op_type = phi::TransToFluidOpName(kernel_pair.first);
            std::vector<std::string> kernel_types;
            for (auto &info_pair : kernel_pair.second) {
              framework::OpKernelType kernel_type =
                  framework::TransPhiKernelKeyToOpKernelType(info_pair.first);
              auto kernel_type_str = framework::KernelTypeToString(kernel_type);
              if (all_kernels_info.count(op_type)) {
                if (std::find(all_kernels_info[op_type].begin(),
                              all_kernels_info[op_type].end(),
                              kernel_type_str) ==
                    all_kernels_info[op_type].end()) {
                  all_kernels_info[op_type].emplace_back(kernel_type_str);
                }
              } else {
                kernel_types.emplace_back(kernel_type_str);
              }
            }
            if (!kernel_types.empty()) {
              all_kernels_info.emplace(op_type, kernel_types);
            }
          }
        }

        return all_kernels_info;
      },
      py::arg("lib") = "all",
      R"DOC(
           Return the registered kernels in paddle.

           Args:
               lib[string]: the library, could be 'phi', 'fluid' and 'all'.
           )DOC");

  m.def(
      "_get_registered_phi_kernels",
      [](const std::string &kernel_registered_type) {
        std::unordered_map<std::string, std::vector<std::string>>
            all_kernels_info;
        auto phi_kernels = phi::KernelFactory::Instance().kernels();
        for (auto &kernel_pair : phi_kernels) {
          auto kernel_name = kernel_pair.first;
          std::vector<std::string> kernel_keys;
          for (auto &info_pair : kernel_pair.second) {
            bool get_function_kernel =
                kernel_registered_type == "function" &&
                info_pair.second.GetKernelRegisteredType() ==
                    phi::KernelRegisteredType::FUNCTION;
            bool get_structure_kernel =
                kernel_registered_type == "structure" &&
                info_pair.second.GetKernelRegisteredType() ==
                    phi::KernelRegisteredType::STRUCTURE;
            if (kernel_registered_type == "all" || get_function_kernel ||
                get_structure_kernel) {
              std::ostringstream stream;
              stream << info_pair.first;
              std::string kernel_key_str = stream.str();
              if (all_kernels_info.count(kernel_name)) {
                bool kernel_exist =
                    std::find(all_kernels_info[kernel_name].begin(),
                              all_kernels_info[kernel_name].end(),
                              kernel_key_str) !=
                    all_kernels_info[kernel_name].end();
                if (!kernel_exist) {
                  all_kernels_info[kernel_name].emplace_back(kernel_key_str);
                }
              } else {
                kernel_keys.emplace_back(kernel_key_str);
              }
            }
          }
          if (!kernel_keys.empty()) {
            all_kernels_info.emplace(kernel_name, kernel_keys);
          }
        }

        return all_kernels_info;
      },
      py::arg("kernel_registered_type") = "function",
      R"DOC(
           Return the registered kernels in phi.

           Args:
               kernel_registered_type[string]: the library, could be 'function', 'structure', and 'all'.
           )DOC");

  // NOTE(Aganlengzi): KernelFactory static instance is initialized BEFORE
  // plugins are loaded for custom kernels, but de-initialized AFTER they are
  // unloaded. We need manually clear symbols(may contain plugins' symbols)
  // stored in this static instance to avoid illegal memory access.
  m.def("clear_kernel_factory",
        []() { phi::KernelFactory::Instance().kernels().clear(); });
  m.def("clear_device_manager", []() {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    platform::XCCLCommContext::Release();
    platform::CustomTracer::Release();
    platform::CustomDeviceEventResourcePool::Release();
    platform::CustomDeviceStreamResourcePool::Release();
    phi::DeviceManager::Release();
#endif
  });

  // NOTE(zjl): ctest would load environment variables at the beginning even
  // though we have not `import paddle.base as base`. So we add this API
  // to enable eager deletion mode in unittest.
  m.def("_set_eager_deletion_mode", &paddle::framework::SetEagerDeletionMode);

  m.def("_set_fuse_parameter_group_size",
        &paddle::framework::ir::SetFuseParameterGroupsSize);
  m.def("_set_fuse_parameter_memory_size",
        &paddle::framework::ir::SetFuseParameterMemorySize);

  m.add_object("_cleanup",
               py::capsule([]() { ScopePool::Instance().Clear(); }));

  m.def("_set_paddle_lib_path", &phi::dynload::SetPaddleLibPath);

  m.def("set_current_thread_name", &phi::SetCurrentThreadName);

  m.def("_promote_types_if_complex_exists",
        &paddle::framework::PromoteTypesIfComplexExists);

  py::class_<Variable>(m, "Variable", R"DOC(Variable Class.

All parameter, weight, gradient are variables in Paddle.
)DOC")
      .def(py::init<>())
      .def("is_int", [](const Variable &var) { return var.IsType<int>(); })
      .def("set_int",
           [](Variable &var, int val) -> void { *var.GetMutable<int>() = val; })
      .def("get_int", [](const Variable &var) -> int { return var.Get<int>(); })
      .def("is_float", [](const Variable &var) { return var.IsType<float>(); })
      .def("set_float",
           [](Variable &var, float val) -> void {
             *var.GetMutable<float>() = val;
           })
      .def("get_float",
           [](const Variable &var) -> float { return var.Get<float>(); })
      .def(
          "get_tensor",
          [](Variable &self) -> phi::DenseTensor * {
            return self.GetMutable<phi::DenseTensor>();
          },
          py::return_value_policy::reference)
      .def("get_bytes",
           [](Variable &self) {
             if (self.IsType<String>()) {  // NOLINT
               return py::bytes(*(self.GetMutable<String>()));
             } else {
               return py::bytes(
                   *(self.GetMutable<RawTensor>()->GetMutable<std::string>()));
             }
           })
      .def("set_string_list",
           [](Variable &self, std::vector<std::string> str_list) {
             *self.GetMutable<Strings>() = str_list;
           })
      .def("set_vocab",
           [](Variable &self,
              const std::unordered_map<std::wstring, std::int32_t> &vocab) {
             *self.GetMutable<Vocab>() = vocab;
           })
      .def(
          "get_string_tensor",
          [](Variable &self) { return self.GetMutable<Strings>(); },
          py::return_value_policy::reference)
      .def(
          "get_map_tensor",
          [](Variable &self) { return self.GetMutable<Vocab>(); },
          py::return_value_policy::reference)
      .def(
          "get_selected_rows",
          [](Variable &self) -> phi::SelectedRows * {
            return self.GetMutable<phi::SelectedRows>();
          },
          py::return_value_policy::reference)
      .def(
          "get_dense_tensor_array",
          [](Variable &self) { return self.GetMutable<phi::TensorArray>(); },
          py::return_value_policy::reference)
      .def(
          "get_fetch_list",
          [](Variable &self) { return self.GetMutable<FetchList>(); },
          py::return_value_policy::reference)
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      .def(
          "get_communicator",
          [](Variable &self) -> platform::Communicator * {
            return self.GetMutable<platform::Communicator>();
          },
          py::return_value_policy::reference)
#endif
      .def(
          "get_reader",
          [](Variable &self) -> framework::ReaderHolder * {
            PADDLE_ENFORCE_EQ(self.IsType<framework::ReaderHolder>(),
                              true,
                              common::errors::InvalidArgument(
                                  "The variable is not type of ReaderHolder."));
            return self.GetMutable<framework::ReaderHolder>();
          },
          py::return_value_policy::reference)
      .def(
          "get_scope",
          [](Variable &self) -> Scope * {
            auto scope_vec = self.GetMutable<std::vector<framework::Scope *>>();
            PADDLE_ENFORCE_GT(
                scope_vec->size(),
                0,
                common::errors::InvalidArgument(
                    "The size of scope_vec should be greater than 0"));
            return scope_vec->front();
          },
          py::return_value_policy::reference)
      .def("set_scope", [](Variable &self, Scope &scope) {
        auto scope_vec = self.GetMutable<std::vector<framework::Scope *>>();
        scope_vec->emplace_back(&scope);
      });

  BindReader(&m);

  py::class_<Scope> _Scope(m, "_Scope", R"DOC(
    Scope is an association of a name to Variable. All variables belong to Scope.

    Variables in a parent scope can be retrieved from local scope.

    You need to specify a scope to run a Net, i.e., `exe.Run(&scope)`.
    One net can run in different scopes and update different variable in the
    scope.

    You can create var in a scope and get it from the scope.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> scope = paddle.static.global_scope()
            >>> place = paddle.CPUPlace()
            >>> # create tensor from a scope and set value to it.
            >>> param = scope.var('Param').get_tensor()
            >>> param_array = np.full((10, 12), 5.0).astype("float32")
            >>> param.set(param_array, place)
        )DOC");
  g_framework_scope_pytype = reinterpret_cast<PyTypeObject *>(_Scope.ptr());
  _Scope
      .def("_remove_from_pool",
           [](Scope &self) { ScopePool::Instance().Remove(&self); })
      .def("raw_address",
           [](Scope &self) { return reinterpret_cast<uint64_t>(&self); })
      .def(
          "var",
          [](Scope &self, const std::string &name) -> Variable * {
            return self.Var(name);
          },
          py::arg("name"),
          R"DOC(
           Find or create variable named :code:`name` in the current scope.

           If the variable named :code:`name` does not exist in the
           current scope, the variable would be created. Otherwise,
           return the existing variable.

           Args:
               name (str): the variable name.

           Returns:
               out (core.Variable): the found or created variable.
           )DOC",
          py::return_value_policy::reference)
      .def("find_var",
           &Scope::FindVar,
           py::arg("name"),
           R"DOC(
           Find variable named :code:`name` in the current scope or
           its parent scope. Return None if not found.

           Args:
               name (str): the variable name.

           Returns:
               out (core.Variable|None): the found variable or None.
           )DOC",
           py::return_value_policy::reference)
      .def("size", &Scope::Size)
      .def("local_var_names",
           &Scope::LocalVarNames,
           R"DOC(
          Get all variable names in the current scope.

          Returns:
              List[str]: The list of variable names.
          )DOC",
           py::return_value_policy::reference)
      .def("erase",
           &Scope::EraseVars,
           py::arg("names"),
           R"DOC(
           Find variable named :code:`name` in the current scope or
           its parent scope. Return None if not found.

           Args:
               name (str): the variable names to be erase.

           Returns:
               None
           )DOC",
           py::return_value_policy::reference)
      .def(
          "new_scope",
          [](Scope &self) -> Scope * { return &self.NewScope(); },
          R"DOC(
           Create a new sub-scope of the current scope.

           Returns:
               out (core._Scope): the created sub-scope.
           )DOC",
          py::return_value_policy::reference)
      .def("drop_kids",
           &Scope::DropKids,
           R"DOC(
           Delete all sub-scopes of the current scope.
           )DOC")
      .def("_kids", &Scope::kids)
      .def_property("_can_reused", &Scope::CanReused, &Scope::SetCanReused);

  m.def(
      "Scope",
      []() -> Scope * {
        auto *s = new Scope();
        ScopePool::Instance().Insert(std::unique_ptr<Scope>(s));
        return s;
      },
      R"DOC(
        Create a new scope.

        Returns:
            out (core._Scope): the created scope.
        )DOC",
      py::return_value_policy::reference);

  //! @note: Be careful! PyBind will return std::string as an unicode, not
  //! Python str. If you want a str object, you should cast them in Python.
  m.def("get_all_op_protos", []() -> std::vector<py::bytes> {
    std::vector<py::bytes> ret_values;
    for (auto &iter : OpInfoMap::Instance().map()) {
      auto &info = iter.second;
      if (info.HasOpProtoAndChecker()) {
        std::string str;
        PADDLE_ENFORCE_EQ(
            info.Proto().SerializeToString(&str),
            true,
            common::errors::Fatal(
                "Serialize OpProto Error. This could be a bug of Paddle."));
        ret_values.emplace_back(str);
      }
    }
    return ret_values;
  });
  m.def(
      "get_all_op_names",
      [](const std::string &lib) {
        std::vector<std::string> op_names;
        for (auto &iter : OpInfoMap::Instance().map()) {
          op_names.emplace_back(iter.first);
        }
        if (lib == "phi") {
          std::vector<std::string> ops_with_phi_kernel;
          for (const auto &op_name : op_names) {
            if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(
                    op_name)) {
              ops_with_phi_kernel.emplace_back(op_name);
            }
          }
          return ops_with_phi_kernel;
        } else if (lib == "fluid") {
          std::vector<std::string> ops_with_fluid_kernel;
          auto all_fluid_op_kernels =
              paddle::framework::OperatorWithKernel::AllOpKernels();
          for (const auto &op_name : op_names) {
            if (all_fluid_op_kernels.find(op_name) !=
                all_fluid_op_kernels.end()) {
              ops_with_fluid_kernel.emplace_back(op_name);
            }
          }
          return ops_with_fluid_kernel;
        } else {
          return op_names;
        }
      },
      py::arg("lib") = "all",
      R"DOC(
      Return the operator names in paddle.

      Args:
          lib[string]: the library contains corresponding OpKernel, could be 'phi', 'fluid' and 'all'. Default value is 'all'.
  )DOC");
  m.def("get_op_attrs_default_value",
        [](py::bytes byte_name) -> paddle::framework::AttributeMap {
          std::string op_type = byte_name;
          paddle::framework::AttributeMap res;
          auto info = OpInfoMap::Instance().GetNullable(op_type);
          if (info != nullptr) {
            if (info->HasOpProtoAndChecker()) {
              auto op_checker = info->Checker();
              res = op_checker->GetDefaultAttrsMap();
            }
          }
          return res;
        });
  m.def(
      "get_op_extra_attrs",
      [](const std::string &op_type)
          -> const paddle::framework::AttributeMap & {
        return operators::ExtraInfoUtils::Instance().GetExtraAttrsMap(op_type);
      });
  m.def(
      "get_attribute_type",
      [](const std::string &op_type,
         const std::string &attr_name) -> paddle::framework::proto::AttrType {
        const auto &default_val =
            operators::ExtraInfoUtils::Instance().GetExtraAttrsMap(op_type).at(
                attr_name);
        return static_cast<paddle::framework::proto::AttrType>(
            default_val.index() - 1);
      });
  m.def("_add_skip_comp_ops", &paddle::prim::PrimCommonUtils::AddSkipCompOps);
  m.def("_set_bwd_prim_blacklist",
        &paddle::prim::PrimCommonUtils::SetPrimBackwardBlacklist);
  m.def("_remove_skip_comp_ops",
        &paddle::prim::PrimCommonUtils::RemoveSkipCompOps);
  m.def("get_grad_op_desc",
        [](const OpDesc &op_desc,
           const std::unordered_set<std::string> &no_grad_set,
           const std::vector<BlockDesc *> &grad_sub_block) {
          std::unordered_map<std::string, std::string> grad_to_var;

          auto op_info = framework::OpInfoMap::Instance().Get(op_desc.Type());
          auto grad_op_maker = op_info.GradOpMaker();
          auto grad_comp_op_maker = op_info.CompGradOpMaker();

          if ((grad_op_maker == nullptr) && (grad_comp_op_maker == nullptr)) {
            // Normally, proto_ should not be null, except some special
            // operators, such as LeaklyReluDoubleGrad op.
            std::string type = op_desc.Type();
            PADDLE_THROW(common::errors::NotFound(
                "Neither operator %s's GradOpMaker nor CompGradOpMaker has "
                "been registered.\nPlease check whether (%s) operator has "
                "gradient operator.\nIf not, please set stop_gradient to be "
                "True for its input and output variables using "
                "var.stop_gradient=True.",
                type.c_str(),
                type.c_str()));
          }

          // In PrimEnabled mode, the priority of CompGradOpMaker is greater
          // than GradCompMaker as we need split first-order grad operator into
          // primitive operators for compiler. In PrimDisabled mode, the
          // priority of CompGradOpMaker is less than GradCompMaker for better
          // performance.
          std::vector<std::unique_ptr<OpDesc>> grad_op_descs;
          auto need_skip =
              paddle::prim::PrimCommonUtils::CheckSkipCompOps(op_desc.Type());
          VLOG(3) << "need skip: " << need_skip << std::endl;
          if (paddle::prim::PrimCommonUtils::IsBwdPrimEnabled()) {
            if ((grad_comp_op_maker != nullptr) && (!need_skip)) {
              VLOG(3) << "Prim Flag Open: Running composite grad fun for "
                      << op_desc.Type();
              grad_op_descs = grad_comp_op_maker(op_desc,
                                                 no_grad_set,
                                                 &grad_to_var,
                                                 op_desc.Block(),
                                                 grad_sub_block);
            } else {
              grad_op_descs = grad_op_maker(
                  op_desc, no_grad_set, &grad_to_var, grad_sub_block);
            }
          } else {
            if (grad_op_maker != nullptr) {
              VLOG(6) << "Prim Flag Close: Running origin grad fun for "
                      << op_desc.Type();
              grad_op_descs = grad_op_maker(
                  op_desc, no_grad_set, &grad_to_var, grad_sub_block);
            } else {
              VLOG(6) << "Prim Flag Close: Running composite grad fun for "
                      << op_desc.Type();
              grad_op_descs = grad_comp_op_maker(op_desc,
                                                 no_grad_set,
                                                 &grad_to_var,
                                                 op_desc.Block(),
                                                 grad_sub_block);
            }
          }

          std::vector<OpDesc *> grad_op_desc_ptrs(grad_op_descs.size());
          std::transform(
              grad_op_descs.begin(),
              grad_op_descs.end(),
              grad_op_desc_ptrs.begin(),
              [](std::unique_ptr<OpDesc> &p) { return p.release(); });
          return std::make_pair(grad_op_desc_ptrs, grad_to_var);
        });
  m.def("has_comp_grad_op_maker", [](const std::string op_type) {
    return framework::OpInfoMap::Instance().Get(op_type).HasCompGradOpMaker();
  });
  m.def("has_grad_op_maker", [](const std::string op_type) {
    return framework::OpInfoMap::Instance().Get(op_type).HasGradOpMaker();
  });
  m.def("has_non_empty_grad_op_maker", [](const std::string op_type) {
    return framework::OpInfoMap::Instance()
        .Get(op_type)
        .HasNonEmptyGradOpMaker();
  });
  m.def("has_empty_grad_op_maker", [](const std::string op_type) {
    return framework::OpInfoMap::Instance().Get(op_type).HasEmptyGradOpMaker();
  });
  m.def("has_infer_inplace", [](const std::string op_type) {
    return framework::OpInfoMap::Instance().Get(op_type).HasInferInplace();
  });
  m.def("infer_no_need_buffer_slots",
        [](const std::string op_type,
           const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs) {
          auto infer_func = framework::OpInfoMap::Instance()
                                .Get(op_type)
                                .NoNeedBufferVarsInferer();
          if (infer_func) {
            return infer_func(inputs, outputs, attrs);
          } else {
            std::unordered_set<std::string> empty = {};
            return empty;
          }
        });
  m.def("prune",
        [](const ProgramDesc &origin,
           const std::set<std::string> &feeded_var_names,
           const std::vector<std::array<size_t, 2>> &targets) {
          ProgramDesc prog_with_targets(origin);

          for (const auto &t : targets) {
            prog_with_targets.MutableBlock(t[0])
                ->Op(static_cast<int>(t[1]))
                ->SetIsTarget(true);
          }
          proto::ProgramDesc pruned_desc;
          auto pruned_origin_block_id_map =
              Prune(*prog_with_targets.Proto(), feeded_var_names, &pruned_desc);
          return std::make_tuple(ProgramDesc(pruned_desc),
                                 pruned_origin_block_id_map);
        });
  m.def(
      "prune_backward",
      [](const framework::ProgramDesc &program) {
        return PruneBackward(program);
      },
      R"DOC(
             Prune the backward part of a program, mostly called in
             program.clone(for_test=True).

            Args:
                   program (ProgramDesc): The original program.

             Returns:
                   tuple(ProgramDesc, map<int, int>): The first part is
                   the pruned program desc, and the second part is a map
                   which contains the id pair of pruned block and corresponding
                   origin block.
           )DOC");
  m.def("empty_var_name",
        []() { return std::string(framework::kEmptyVarName); });
  m.def("grad_var_suffix",
        []() { return std::string(framework::kGradVarSuffix); });
  m.def_submodule(
       "var_names",
       "The module will return special predefined variable name in Paddle")
      .def("empty", []() { return kEmptyVarName; })
      .def("temp", []() { return kTempVarName; });

  py::class_<phi::DeviceContext>(m, "DeviceContext")
      .def_static("create",
                  [](phi::CPUPlace &place) -> phi::DeviceContext * {
                    auto *context = new phi::CPUContext();
                    context->SetAllocator(
                        paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetAllocator(place)
                            .get());
                    context->SetHostAllocator(
                        paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetAllocator(phi::CPUPlace())
                            .get());
                    context->SetZeroAllocator(
                        paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetZeroAllocator(place)
                            .get());
                    context->SetHostZeroAllocator(
                        paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetZeroAllocator(phi::CPUPlace())
                            .get());
                    return context;
                  })
      .def_static(
          "create",
          [](phi::XPUPlace &place) -> phi::DeviceContext * {
#ifndef PADDLE_WITH_XPU
            PADDLE_THROW(common::errors::PermissionDenied(
                "Cannot use XPUPlace in CPU/GPU version, "
                "Please recompile or reinstall Paddle with XPU support."));
#else
            auto *context = new phi::XPUContext(place);
            context->SetAllocator(
                paddle::memory::allocation::AllocatorFacade::Instance()
                    .GetAllocator(place)
                    .get());
            context->SetHostAllocator(
                paddle::memory::allocation::AllocatorFacade::Instance()
                    .GetAllocator(phi::CPUPlace())
                    .get());
            context->SetZeroAllocator(
                paddle::memory::allocation::AllocatorFacade::Instance()
                    .GetZeroAllocator(place)
                    .get());
            context->SetHostZeroAllocator(
                paddle::memory::allocation::AllocatorFacade::Instance()
                    .GetZeroAllocator(phi::CPUPlace())
                    .get());
            return context;
#endif
          })
      .def_static("create",
                  [](phi::CustomPlace &place) -> phi::DeviceContext * {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
                    PADDLE_THROW(common::errors::PermissionDenied(
                        "Cannot use CustomPlace in CPU/GPU/XPU version, "
                        "Please recompile or reinstall Paddle with "
                        "CustomDevice support."));
#else
            return new phi::CustomContext(place);
#endif
                  })
      .def_static(
          "create",
          [](phi::GPUPlace &place) -> phi::DeviceContext * {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
            PADDLE_THROW(common::errors::PermissionDenied(
                "Cannot use CUDAPlace in CPU only version, "
                "Please recompile or reinstall Paddle with CUDA support."));
#else
            auto *context = new phi::GPUContext(place);
            context->SetAllocator(
                paddle::memory::allocation::AllocatorFacade::Instance()
                    .GetAllocator(place, context->stream())
                    .get());
            context->SetHostAllocator(
                paddle::memory::allocation::AllocatorFacade::Instance()
                    .GetAllocator(phi::CPUPlace())
                    .get());
            context->SetZeroAllocator(
                paddle::memory::allocation::AllocatorFacade::Instance()
                    .GetZeroAllocator(place)
                    .get());
            context->SetHostZeroAllocator(
                paddle::memory::allocation::AllocatorFacade::Instance()
                    .GetZeroAllocator(phi::CPUPlace())
                    .get());
            context->SetPinnedAllocator(
                paddle::memory::allocation::AllocatorFacade::Instance()
                    .GetAllocator(phi::GPUPinnedPlace())
                    .get());
            context->PartialInitWithAllocator();
            return context;
#endif
          })
      .def_static(
          "create", [](phi::GPUPinnedPlace &place) -> phi::DeviceContext * {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
            PADDLE_THROW(common::errors::PermissionDenied(
                "Cannot use CUDAPinnedPlace in CPU only version, "
                "Please recompile or reinstall Paddle with CUDA support."));
#else
            return new phi::GPUPinnedContext(place);
#endif
          });
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  py::class_<platform::Communicator>(m, "Communicator").def(py::init<>());
#endif
  m.def("get_all_device_type", []() {
    std::vector<std::string> device_types;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    device_types = phi::DeviceManager::GetAllDeviceTypes();
#else
          VLOG(1) << string::Sprintf(
              "Cannot use get_all_device_type because you have installed "
              "CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_all_device_type, please try to install"
              "CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle\n");
#endif
    return device_types;
  });
  m.def("get_all_custom_device_type", []() {
    std::vector<std::string> device_types;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
#else
          VLOG(1) << string::Sprintf(
              "Cannot use get_all_custom_device_type because you have "
              "installed CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_all_custom_device_type, please try to "
              "install CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle\n");
#endif
    return device_types;
  });
  m.def("get_available_device", [] {
    std::vector<std::string> devices;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    devices = phi::DeviceManager::GetAllDeviceList();
#else
          VLOG(1) << string::Sprintf(
              "Cannot use get_available_device because you have installed "
              "CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_available_device, please try to install"
              "CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle\n");
#endif
    return devices;
  });
  m.def("get_available_custom_device", [] {
    std::vector<std::string> devices;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    devices = phi::DeviceManager::GetAllCustomDeviceList();
#else
          VLOG(1) << string::Sprintf(
              "Cannot use get_available_custom_device because you have "
              "installed CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_available_custom_device, please try to "
              "install"
              "CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle\n");
#endif
    return devices;
  });
  m.def("get_custom_device_count", [](const std::string &device_type) {
    size_t device_count = 0;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    // TODO(duanyanhui): Optimize DeviceManager::GetDeviceCount to support
    // returning default device when only one device is registered in
    // DeviceManager.
    device_count = phi::DeviceManager::GetDeviceCount(device_type);
#else
          VLOG(1) << string::Sprintf(
              "Cannot use get_custom_device_count because you have "
              "installed CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_custom_device_count, please try to "
              "install"
              "CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle\n");
#endif
    return device_count;
  });

  py::class_<OperatorBase>(m, "Operator")
      .def_static("create",
                  [](py::bytes protobin) {
                    proto::OpDesc desc;
                    PADDLE_ENFORCE_EQ(desc.ParsePartialFromString(protobin),
                                      true,
                                      common::errors::InvalidArgument(
                                          "Cannot parse user input to OpDesc"));
                    PADDLE_ENFORCE_EQ(desc.IsInitialized(),
                                      true,
                                      common::errors::InvalidArgument(
                                          "The provided OpDesc is not "
                                          "initialized, the reason is: %s",
                                          desc.InitializationErrorString()));
                    return OpRegistry::CreateOp(desc);
                  })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const phi::CPUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const phi::XPUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const phi::GPUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const phi::GPUPinnedPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const phi::CustomPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("type",
           [](const OperatorBase &op) -> std::string { return op.Type(); })
      .def("outputs",
           [](const OperatorBase &op)
               -> std::map<std::string, std::vector<std::string>> {
             return op.Outputs();
           })
      .def("output_vars",
           [](const OperatorBase &op) { return op.OutputVars(true); })
      .def("inputs", [](const OperatorBase &op) { return op.Inputs(); })
      .def("input_vars", [](const OperatorBase &op) { return op.InputVars(); })
      .def("__str__", &OperatorBase::DebugString)
      .def("no_intermediate_outputs",
           [](const OperatorBase &op) { return op.OutputVars(false); })
      .def("support_gpu", &OperatorBase::SupportGPU);

  py::class_<framework::ExecutorPrepareContext>(m, "ExecutorPrepareContext")
      .def(py::init<const ProgramDesc &, size_t>());

  py::class_<framework::TrainerBase, std::shared_ptr<framework::TrainerBase>>(
      m, "TrainerBase")
      .def(
          "get_worker_scope",
          [](TrainerBase &self, int thread_id) -> Scope * {
            return self.GetWorkerScope(thread_id);
          },
          py::return_value_policy::reference)
      .def("finalize", &TrainerBase::Finalize)
      .def("ResetDataset", &TrainerBase::ResetDataset);

  m.def("_get_eager_deletion_vars", &framework::GetEagerDeletionCleanVars);

  py::class_<framework::Executor>(m, "Executor")
      .def(py::init<const phi::Place &>())
      .def("close", &Executor::Close)
      .def("get_place", &Executor::GetPlace)
      .def("run_from_dataset",
           &Executor::RunFromDataset,
           py::call_guard<py::gil_scoped_release>())
      .def("release_trainer",
           &Executor::ReleaseTrainer,
           py::call_guard<py::gil_scoped_release>())
      .def("init_for_dataset",
           [](Executor &self,
              const ProgramDesc &prog,
              const std::string &trainer_desc,
              Scope *scope,
              Dataset *dataset) -> std::shared_ptr<TrainerBase> {
             pybind11::gil_scoped_release release;
             return self.InitForDataset(prog, trainer_desc, scope, dataset);
           })
      .def("run_from_dataset",
           [](Executor &self, std::shared_ptr<TrainerBase> trainer) {
             pybind11::gil_scoped_release release;
             self.RunFromDataset(trainer);
           })
      .def("run_prepared_ctx",
           [](Executor &self,
              ExecutorPrepareContext *ctx,
              Scope *scope,
              std::map<std::string, const phi::DenseTensor *> *feed_targets,
              std::map<std::string, FetchType *> *fetch_targets,
              bool create_local_scope = true,
              bool create_vars = true,
              const std::string &feed_holder_name = "feed",
              const std::string &fetch_holder_name = "fetch") {
             pybind11::gil_scoped_release release;
             self.RunPreparedContext(ctx,
                                     scope,
                                     feed_targets,
                                     fetch_targets,
                                     create_local_scope,
                                     create_vars,
                                     feed_holder_name,
                                     fetch_holder_name);
           })
      .def("run_prepared_ctx",
           [](Executor &self,
              ExecutorPrepareContext *ctx,
              Scope *scope,
              bool create_local_scope = true,
              bool create_vars = true,
              bool keep_kids = false) {
             pybind11::gil_scoped_release release;
             self.RunPreparedContext(
                 ctx, scope, create_local_scope, create_vars, keep_kids);
           })
      .def("prepare",
           [](Executor &self,
              const ProgramDesc &program,
              int block_id,
              const std::vector<std::string> &skip_ref_cnt_vars =
                  std::vector<std::string>(),
              bool force_disable_gc = false) {
             pybind11::gil_scoped_release release;
             return self.Prepare(
                 program, block_id, skip_ref_cnt_vars, force_disable_gc);
           })
      .def("create_variables", &Executor::CreateVariables)
      .def("run",
           [](Executor &self,
              const ProgramDesc &prog,
              Scope *scope,
              int block_id,
              bool create_local_scope,
              bool create_vars,
              const std::vector<std::string> &fetch_vars) {
             pybind11::gil_scoped_release release;
             self.Run(prog,
                      scope,
                      block_id,
                      create_local_scope,
                      create_vars,
                      fetch_vars);
           });

  py::class_<framework::interpreter::CostInfo>(m, "CostInfo")
      .def(py::init<>())
      .def("total_time",
           [](interpreter::CostInfo &self) { return self.total_time; })
      .def("device_memory_bytes", [](interpreter::CostInfo &self) {
        return self.device_memory_bytes;
      });

  py::class_<framework::StandaloneExecutor>(m, "StandaloneExecutor")
      .def(py::init<const phi::Place &, const interpreter::Plan &, Scope *>())
      .def("run",
           [](StandaloneExecutor &self,
              std::vector<std::string> feed_names,
              bool enable_job_schedule_profiler = false) {
             paddle::framework::FetchList ret;
             {
               pybind11::gil_scoped_release release;
               ret = self.Run(feed_names, enable_job_schedule_profiler);
             }
             return py::cast(std::move(ret));
           })
      .def("run_profile",
           [](StandaloneExecutor &self, std::vector<std::string> feed_names) {
             std::shared_ptr<framework::ProgramDesc> program_desc;
             {
               pybind11::gil_scoped_release release;
               program_desc = self.RunProfile(feed_names);
             }
             return py::cast(std::move(program_desc));
           });

  py::class_<framework::interpreter::Job,
             std::shared_ptr<framework::interpreter::Job>>(m, "Job")
      .def(py::init<const std::string &>(), py::arg("type"))
      .def("micro_batch_id", &framework::interpreter::Job::MicroBatchId)
      .def("type", &framework::interpreter::Job::Type)
      .def("set_micro_batch_id", &framework::interpreter::Job::SetMicroBatchId)
      .def("set_skip_gc_vars", &framework::interpreter::Job::SetSkipGcVars);

  py::class_<framework::interpreter::Plan>(m, "Plan")
      .def(
          py::init<
              const std::vector<std::shared_ptr<framework::interpreter::Job>> &,
              const std::unordered_map<std::string,
                                       std::shared_ptr<framework::ProgramDesc>>
                  &>(),
          py::arg("job_list"),
          py::arg("type_to_program"))
      .def(
          py::init<
              const std::vector<std::shared_ptr<framework::interpreter::Job>> &,
              const std::unordered_map<std::string,
                                       std::shared_ptr<::pir::Program>> &>(),
          py::arg("job_list"),
          py::arg("type_to_ir_program"))
      .def("job_list", &framework::interpreter::Plan::JobList)
      .def("job_types", &framework::interpreter::Plan::JobTypes)
      .def("micro_batch_num", &framework::interpreter::Plan::MicroBatchNum)
      .def("set_ir_program", &framework::interpreter::Plan::SetIrProgram)
      .def("ir_program", &framework::interpreter::Plan::IrProgram)
      .def("program", &framework::interpreter::Plan::Program);

  m.def("get_no_need_buffer_values",
        framework::interpreter::GetNoNeedBufferValues);
#ifdef PADDLE_WITH_CUDA
  py::class_<phi::GPUEventTimer>(m, "GPUEventTimer")
      .def(py::init<phi::GPUPlace>(), py::arg("place"))
      .def(
          "start",
          [](phi::GPUEventTimer &timer) { timer.Start(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "stop",
          [](phi::GPUEventTimer &timer) { timer.Stop(); },
          py::call_guard<py::gil_scoped_release>())
      .def("reset",
           &phi::GPUEventTimer::Reset,
           py::call_guard<py::gil_scoped_release>())
      .def("elapsed",
           &phi::GPUEventTimer::Elapsed,
           py::arg("reset") = true,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "elapsed_list",
          [](phi::GPUEventTimer &timer, bool reset) {
            std::vector<double> values;
            {
              py::gil_scoped_release release;
              values = timer.ElapsedList(reset);
            }
            size_t n = values.size();
            py::array_t<double, py::array::c_style | py::array::forcecast>
                array(n);
            auto buffer = array.request();
            std::memcpy(buffer.ptr, values.data(), sizeof(values[0]) * n);
            return array;
          },
          py::arg("reset") = true)
      .def("pre_alloc",
           &phi::GPUEventTimer::PreAlloc,
           py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("shrink_to_fit",
           &phi::GPUEventTimer::ShrinkToFit,
           py::call_guard<py::gil_scoped_release>())
      .def("size",
           &phi::GPUEventTimer::Size,
           py::call_guard<py::gil_scoped_release>())
      .def("capacity",
           &phi::GPUEventTimer::Capacity,
           py::call_guard<py::gil_scoped_release>());
#endif

  m.def("init_gflags", framework::InitGflags);
  m.def("init_glog", framework::InitGLOG);
  m.def("init_memory_method", framework::InitMemoryMethod);
  m.def("load_op_meta_info_and_register_op", [](const std::string dso_name) {
    egr::Controller::Instance().MergeOpMetaInfoMap(
        framework::LoadOpMetaInfoAndRegisterOp(dso_name));
  });
  m.def("init_devices", []() {
    framework::InitDevices();
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    for (auto &dev_type : phi::DeviceManager::GetAllCustomDeviceTypes()) {
      paddle::operators::RegisterCustomDeviceCommonKernel(dev_type);
    }
#endif
  });
  m.def("init_default_kernel_signatures",
        []() { framework::InitDefaultKernelSignatureMap(); });
  m.def("init_tensor_operants", []() {
    paddle::OperantsManager::Instance().eager_operants =
        std::make_unique<paddle::prim::EagerTensorOperants>();
    paddle::OperantsManager::Instance().static_operants =
        std::make_unique<paddle::prim::StaticTensorOperants>();
    paddle::OperantsManager::Instance().phi_operants =
        std::make_unique<paddle::operants::PhiTensorOperants>();
    VLOG(4) << "Initialize tensor operants successfully";
  });
  m.def("is_compiled_with_avx", IsCompiledWithAVX);
  m.def("is_compiled_with_cuda", IsCompiledWithCUDA);
  m.def("is_compiled_with_cudnn_frontend", IsCompiledWithCudnnFrontend);
  m.def("is_compiled_with_rocm", IsCompiledWithROCM);
  m.def("is_compiled_with_custom_device", IsCompiledWithCustomDevice);
  m.def("is_compiled_with_ipu", IsCompiledWithIPU);
  m.def("is_compiled_with_xpu", IsCompiledWithXPU);
  m.def("is_compiled_with_mkldnn", IsCompiledWithMKLDNN);
  m.def("is_compiled_with_nccl", IsCompiledWithNCCL);
  m.def("is_compiled_with_mpi", IsCompiledWithMPI);
  m.def("is_compiled_with_mpi_aware", IsCompiledWithMPIAWARE);
  m.def("is_compiled_with_cinn", IsCompiledWithCINN);
  m.def("is_compiled_with_distribute", IsCompiledWithDISTRIBUTE);
  m.def("_is_compiled_with_heterps", IsCompiledWithHETERPS);
  m.def("supports_bfloat16", SupportsBfloat16);
  m.def("supports_bfloat16_fast_performance", SupportsBfloat16FastPerformance);
  m.def("supports_int8", SupportsInt8);
  m.def("supports_avx512f", SupportsAvx512F);
  m.def("supports_vnni", SupportsVNNI);
  m.def("op_supported_infos", imperative::OpSupportedInfos);
  m.def("is_compiled_with_brpc", IsCompiledWithBrpc);
  m.def("is_compiled_with_dist", IsCompiledWithDIST);
  m.def("_cuda_synchronize", [](const phi::GPUPlace &place) {
    phi::DeviceContextPool::Instance().Get(place)->Wait();
  });
  m.def("_set_warmup", [](bool warmup) {
#if defined(PADDLE_WITH_CUDA)
    paddle::memory::allocation::AutoGrowthBestFitAllocatorV2State::GetInstance()
        .SetWarmup(warmup);
#endif
  });
  m.def("_test_enforce_gpu_success", []() {
#if defined(PADDLE_WITH_CUDA)
    PADDLE_ENFORCE_GPU_SUCCESS(cudaErrorInsufficientDriver);
#endif
  });

  m.def("get_float_stats", []() {
    std::vector<paddle::platform::ExportedStatValue<float>> float_stats;
    paddle::platform::StatRegistry<float>::Instance().publish(float_stats);
    std::unordered_map<std::string, float> stats_map;
    for (const auto &stat : float_stats) {
      stats_map[stat.key] = stat.value;
    }
    return stats_map;
  });
  m.def("get_int_stats", []() {
    std::vector<paddle::platform::ExportedStatValue<int64_t>> int_stats;
    paddle::platform::StatRegistry<int64_t>::Instance().publish(int_stats);
    std::unordered_map<std::string, int64_t> stats_map;
    for (const auto &stat : int_stats) {
      stats_map[stat.key] = stat.value;
    }
    return stats_map;
  });
  m.def("device_memory_stat_current_value",
        memory::DeviceMemoryStatCurrentValue);
  m.def("device_memory_stat_peak_value", memory::DeviceMemoryStatPeakValue);
  m.def("device_memory_stat_reset_peak_value",
        memory::DeviceMemoryStatResetPeakValue);

  m.def("host_memory_stat_current_value", memory::HostMemoryStatCurrentValue);
  m.def("host_memory_stat_peak_value", memory::HostMemoryStatPeakValue);
  m.def("host_memory_stat_reset_peak_value",
        memory::HostMemoryStatResetPeakValue);
  m.def(
      "run_cmd",
      [](const std::string &cmd,
         int time_out = -1,
         int sleep_inter = -1) -> const std::string {
        return paddle::framework::shell_get_command_output(
            cmd, time_out, sleep_inter);
      },
      py::arg("cmd"),
      py::arg("time_out") = -1,
      py::arg("sleep_inter") = -1);
  m.def(
      "shell_execute_cmd",
      [](const std::string &cmd,
         int time_out = 0,
         int sleep_inter = 0,
         bool redirect_stderr = false) -> std::vector<std::string> {
        return paddle::framework::shell_execute_cmd(
            cmd, time_out, sleep_inter, redirect_stderr);
      },
      py::arg("cmd"),
      py::arg("time_out") = 0,
      py::arg("sleep_inter") = 0,
      py::arg("redirect_stderr") = false);
  m.def("set_variable",
        static_cast<void (*)(  // NOLINT
            Scope *,
            const phi::DenseTensor &,
            const std::string &)>(&framework::SetVariable));
  m.def("set_feed_variable",
        static_cast<void (*)(  // NOLINT
            Scope *,
            const phi::DenseTensor &,
            const std::string &,
            size_t)>(&framework::SetFeedVariable));
  m.def("get_fetch_variable",
        [](const Scope &scope,
           const std::string &var_name,
           size_t index) -> py::object {
          auto &var = framework::GetFetchVariable(scope, var_name, index);
          if (data_is_dense_tensor(var)) {  // NOLINT
            return py::cast(PADDLE_GET(phi::DenseTensor, var));
          } else {
            return py::cast(PADDLE_GET(phi::TensorArray, var));
          }
        });
  m.def("get_variable_tensor", framework::GetVariableTensor);

  m.def("_is_program_version_supported", IsProgramVersionSupported);
#if defined(PADDLE_WITH_CUDA)
  m.def("allocator_dump", [](const phi::GPUPlace &place) {
    auto allocator = std::dynamic_pointer_cast<
        paddle::memory::allocation::AutoGrowthBestFitAllocator>(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAutoGrowthAllocator(place));
    allocator->DumpInfo();
  });
#endif
  BindProgramDesc(&m);
  BindBlockDesc(&m);
  BindVarDesc(&m);
  BindOpDesc(&m);
  BindCostModel(&m);
  BindConstValue(&m);
  BindGlobalValueGetterSetter(&m);
  BindTCPStore(&m);
  BindCommContextManager(&m);
  BindAutoParallel(&m);
  BindJitProperty(&m);

  py::class_<phi::TensorArray> pydensetensorarray(m, "DenseTensorArray", R"DOC(
    DenseTensorArray is array of DenseTensor, it supports operator[], len() and for-loop iteration.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> arr = paddle.framework.core.DenseTensorArray()
)DOC");
  g_framework_densetensorarray_pytype =
      reinterpret_cast<PyTypeObject *>(pydensetensorarray.ptr());
  pydensetensorarray
      .def(py::init([]() { return std::make_unique<phi::TensorArray>(); }))
      .def(
          "__getitem__",
          [](phi::TensorArray &self, size_t i) { return &self.at(i); },
          py::return_value_policy::reference)
      .def("__len__", [](phi::TensorArray &self) { return self.size(); })
      .def("__setitem__",
           [](phi::TensorArray &self, size_t i, const phi::DenseTensor &t) {
             PADDLE_ENFORCE_LT(i,
                               self.size(),
                               common::errors::InvalidArgument(
                                   "The index to set is larger than the size "
                                   "of DenseTensorArray."));
             self[i].ShareDataWith(t);
           })
      .def(
          "append",
          [](phi::TensorArray &self, const phi::DenseTensor &t) {
            self.emplace_back();
            self.back().ShareDataWith(t);
            self.back().set_lod(t.lod());
          },
          py::arg("tensor"),
          R"DOC(
             Append a DenseTensor to DenseTensorArray.

             Args:
                   tensor (DenseTensor): The DenseTensor to be appended.

             Returns:
                   None.

             Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import numpy as np

                        >>> arr = paddle.framework.core.DenseTensorArray()
                        >>> t = paddle.framework.core.DenseTensor()
                        >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
                        >>> arr.append(t)
           )DOC")
      .def(
          "_move_to_list",
          [](phi::TensorArray &self) -> py::list {
            py::list res(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
              res[i] = py::cast(std::move(self[i]));
            }
            self.clear();
            return res;
          },
          py::return_value_policy::take_ownership);

  py::class_<FetchList>(m, "FetchList", R"DOC( FetchList is a
        vector of paddle::variant<DenseTensor, DenseTensorArray>.
        )DOC")
      .def(
          "_move_to_list",
          [](FetchList &self) -> py::list {
            py::list res(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
              if (data_is_dense_tensor(self[i])) {
                auto &data = PADDLE_GET(phi::DenseTensor, self[i]);
                res[i] = py::cast(std::move(data));
              } else if (data_is_sparse_coo_tensor(self[i])) {
                auto &data = PADDLE_GET(phi::SparseCooTensor, self[i]);
                res[i] = py::cast(std::move(data));
              } else {
                auto &data = PADDLE_GET(phi::TensorArray, self[i]);
                py::list tmp(data.size());
                for (size_t j = 0; j < data.size(); ++j) {
                  tmp[j] = py::cast(std::move(data[j]));
                }
                res[i] = std::move(tmp);
              }
            }
            self.clear();
            return res;
          },
          py::return_value_policy::take_ownership)

      .def(
          "append",
          [](FetchList &self, const phi::DenseTensor &t) {
            self.emplace_back();
            auto &dense_tensor = PADDLE_GET(phi::DenseTensor, self.back());
            dense_tensor.ShareDataWith(t);
          },
          py::arg("var"))

      .def(
          "append",
          [](FetchList &self, const phi::TensorArray &t) {
            self.emplace_back();
            auto &dense_tensor_array =
                PADDLE_GET(phi::TensorArray, self.back());
            for (size_t i = 0; i < t.size(); ++i) {
              dense_tensor_array[i].ShareDataWith(t[i]);
            }
          },
          py::arg("var"));

  py::class_<FetchUnmergedList>(m, "FetchUnmergedList", R"DOC(
        FetchUnmergedList is 2-D array of FetchType(paddle::variant(DenseTensor, DenseTensorArray)).
        )DOC")
      .def(
          "_move_to_list",
          [](FetchUnmergedList &self) -> py::list {
            py::list res(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
              py::list tmp(self[i].size());
              for (size_t j = 0; j < self[i].size(); ++j) {
                if (data_is_dense_tensor(self[i][j])) {
                  auto &var = PADDLE_GET(phi::DenseTensor, self[i][j]);
                  tmp[j] = py::cast(std::move(var));
                } else {
                  auto &var = PADDLE_GET(phi::TensorArray, self[i][j]);
                  py::list tmp_array(var.size());
                  for (size_t k = 0; k < var.size(); ++k) {
                    tmp_array[k] = std::move(var[k]);
                  }
                  tmp[j] = std::move(tmp_array);
                }
              }
              res[i] = std::move(tmp);
              self[i].clear();
            }
            self.clear();
            return res;
          },
          py::return_value_policy::take_ownership);

  m.def("op_support_gpu", OpSupportGPU);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  m.def("get_cuda_device_count", platform::GetGPUDeviceCount);
  m.def("get_cuda_current_device_id", &platform::GetCurrentDeviceId);
  m.def("set_cuda_current_device_id", &platform::SetDeviceId, py::arg("i"));
  m.def("cuda_empty_cache", [] {
    for (int dev_id : platform::GetSelectedDevices()) {
      auto *dev_ctx =
          phi::DeviceContextPool::Instance().GetByPlace(phi::GPUPlace(dev_id));
      dev_ctx->cudnn_workspace_handle().ResetWorkspace();
    }
    platform::EmptyCache();
  });
  m.def(
      "get_device_properties",
      [](int id) -> const gpuDeviceProp & {
        return platform::GetDeviceProperties(id);
      },
      py::return_value_policy::copy);

  py::class_<gpuDeviceProp>(m, "_gpuDeviceProperties", py::module_local())
      .def_property_readonly(
          "name", [](const gpuDeviceProp &prop) { return prop.name; })
      .def_property_readonly(
          "major", [](const gpuDeviceProp &prop) { return prop.major; })
      .def_property_readonly(
          "minor", [](const gpuDeviceProp &prop) { return prop.minor; })
      .def_property_readonly(
          "total_memory",
          [](const gpuDeviceProp &prop) { return prop.totalGlobalMem; })
      .def_property_readonly(
          "multi_processor_count",
          [](const gpuDeviceProp &prop) { return prop.multiProcessorCount; })
      .def_property_readonly(
          "is_multi_gpu_board",
          [](const gpuDeviceProp &prop) { return prop.isMultiGpuBoard; })
      .def_property_readonly(
          "is_integrated",
          [](const gpuDeviceProp &prop) { return prop.integrated; })
      .def("__repr__", [](const gpuDeviceProp &prop) {
        std::stringstream ostr;
        ostr << "_gpuDeviceProperties(name='" << prop.name
             << "', major=" << prop.major << ", minor=" << prop.minor
             << ", total_memory=" << prop.totalGlobalMem / (1024 * 1024)
             << "MB, multi_processor_count=" << prop.multiProcessorCount << ")";
        return ostr.str();
      });

#if !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
  m.def("nvprof_init", platform::CudaProfilerInit);
  m.def("nvprof_start", platform::CudaProfilerStart);
  m.def("nvprof_stop", platform::CudaProfilerStop);
  m.def("nvprof_nvtx_push", [](const std::string &name) {
    platform::CudaNvtxRangePush(name, platform::NvtxRangeColor::Green);
  });
  m.def("nvprof_nvtx_pop", platform::CudaNvtxRangePop);
  m.def("nvprof_enable_record_event", platform::NvprofEnableRecordEvent);
  m.def("nvprof_disable_record_event", platform::NvprofDisableRecordEvent);
#endif
#endif

#ifdef PADDLE_WITH_IPU
  m.def("get_ipu_device_count", platform::GetIPUDeviceCount);
#endif

#ifdef PADDLE_WITH_XPU
  m.def("get_xpu_device_count", platform::GetXPUDeviceCount);
  m.def("xpu_empty_cache", platform::EmptyCache);
#endif

  py::enum_<platform::TracerOption>(m, "TracerOption", py::arithmetic())
      .value("kDefault", platform::TracerOption::kDefault)
      .value("kOpDetail", platform::TracerOption::kOpDetail)
      .value("kAllOpDetail", platform::TracerOption::kAllOpDetail)
      .export_values();

  py::enum_<platform::ProfilerState>(m, "ProfilerState", py::arithmetic())
      .value("kDisabled", platform::ProfilerState::kDisabled)
      .value("kCPU", platform::ProfilerState::kCPU)
      .value("kCUDA", platform::ProfilerState::kCUDA)
      .value("kAll", platform::ProfilerState::kAll)
      .export_values();

  py::enum_<platform::EventSortingKey>(m, "EventSortingKey", py::arithmetic())
      .value("kDefault", platform::EventSortingKey::kDefault)
      .value("kCalls", platform::EventSortingKey::kCalls)
      .value("kTotal", platform::EventSortingKey::kTotal)
      .value("kMin", platform::EventSortingKey::kMin)
      .value("kMax", platform::EventSortingKey::kMax)
      .value("kAve", platform::EventSortingKey::kAve)
      .export_values();

  m.def("set_tracer_option", platform::SetTracerOption);
  m.def("enable_profiler", platform::EnableProfiler);
  m.def("disable_profiler", platform::DisableProfiler);
  m.def("is_profiler_enabled", platform::IsProfileEnabled);
  m.def("reset_profiler", platform::ResetProfiler);
  m.def("register_pass", [](const std::string &pass_type, py::object callable) {
    PADDLE_ENFORCE_EQ(
        framework::ir::PassRegistry::Instance().Has(pass_type),
        false,
        common::errors::AlreadyExists("Pass '%s' is registered more than "
                                      "once. Please use another name.",
                                      pass_type));
    callable.inc_ref();
    framework::ir::PassRegistry::Instance().Insert(
        pass_type, [pass_type, callable]() {
          py::gil_scoped_acquire guard;
          std::unique_ptr<framework::ir::Pass> pass(
              new framework::ir::GeneratePass(py::cast<std::string>(callable()),
                                              pass_type));
          return pass;
        });
  });
  m.def("get_pass", [](const std::string &pass_type) {
    auto pass = framework::ir::PassRegistry::Instance().Get(pass_type);
    return std::shared_ptr<framework::ir::Pass>(std::move(pass));
  });
  m.def("register_subgraph_pass", [](const std::string &pass_type) {
    framework::ir::Pass::AddSupportSubgraphPass(pass_type);
  });

  m.def("size_of_dtype", framework::SizeOfType);
  m.def("size_of_dtype", phi::SizeOf);
  py::class_<paddle::platform::ProfilerResult>(m, "_ProfilerResult")
      .def(py::init<>())
      .def("get_data",
           &paddle::platform::ProfilerResult::GetData,
           py::return_value_policy::automatic_reference)
      .def("save", &paddle::platform::ProfilerResult::Save)
      .def("get_extra_info", &paddle::platform::ProfilerResult::GetExtraInfo)
      .def("get_version", &paddle::platform::ProfilerResult::GetVersion)
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def("get_span_indx", &paddle::platform::ProfilerResult::GetSpanIndx)
      .def("get_device_property",
           &paddle::platform::ProfilerResult::GetDeviceProperty);
#else
      .def("get_span_indx", &paddle::platform::ProfilerResult::GetSpanIndx);
#endif

  py::class_<paddle::platform::MemPythonNode>(m, "MemPythonNode")
      .def(py::init<>())
      .def_readwrite("timestamp_ns",
                     &paddle::platform::MemPythonNode::timestamp_ns)
      .def_readwrite("addr", &paddle::platform::MemPythonNode::addr)
      .def_readwrite("type", &paddle::platform::MemPythonNode::type)
      .def_readwrite("process_id", &paddle::platform::MemPythonNode::process_id)
      .def_readwrite("thread_id", &paddle::platform::MemPythonNode::thread_id)
      .def_readwrite("increase_bytes",
                     &paddle::platform::MemPythonNode::increase_bytes)
      .def_readwrite("place", &paddle::platform::MemPythonNode::place)
      .def_readwrite("current_allocated",
                     &paddle::platform::MemPythonNode::current_allocated)
      .def_readwrite("current_reserved",
                     &paddle::platform::MemPythonNode::current_reserved)
      .def_readwrite("peak_allocated",
                     &paddle::platform::MemPythonNode::peak_allocated)
      .def_readwrite("peak_reserved",
                     &paddle::platform::MemPythonNode::peak_reserved)
      .def("__repr__", [](paddle::platform::MemPythonNode &event_node) {
        std::stringstream ostr;
        ostr << "MemPythonNode(timestamp_ns=" << event_node.timestamp_ns
             << ", addr=" << event_node.addr << ", type='"
             << paddle::platform::StringTracerMemEventType(event_node.type)
             << "', process_id=" << event_node.process_id
             << ", thread_id=" << event_node.thread_id << ")";
        return ostr.str();
      });

  py::class_<paddle::platform::DevicePythonNode>(m, "DevicePythonNode")
      .def(py::init<>())
      .def_readwrite("name", &paddle::platform::DevicePythonNode::name)
      .def_readwrite("type", &paddle::platform::DevicePythonNode::type)
      .def_readwrite("start_ns", &paddle::platform::DevicePythonNode::start_ns)
      .def_readwrite("end_ns", &paddle::platform::DevicePythonNode::end_ns)
      .def_readwrite("device_id",
                     &paddle::platform::DevicePythonNode::device_id)
      .def_readwrite("context_id",
                     &paddle::platform::DevicePythonNode::context_id)
      .def_readwrite("stream_id",
                     &paddle::platform::DevicePythonNode::stream_id)
      .def_readwrite("correlation_id",
                     &paddle::platform::DevicePythonNode::correlation_id)
      .def_readwrite("block_x", &paddle::platform::DevicePythonNode::block_x)
      .def_readwrite("block_y", &paddle::platform::DevicePythonNode::block_y)
      .def_readwrite("block_z", &paddle::platform::DevicePythonNode::block_z)
      .def_readwrite("grid_x", &paddle::platform::DevicePythonNode::grid_x)
      .def_readwrite("grid_y", &paddle::platform::DevicePythonNode::grid_y)
      .def_readwrite("grid_z", &paddle::platform::DevicePythonNode::grid_z)
      .def_readwrite("shared_memory",
                     &paddle::platform::DevicePythonNode::shared_memory)
      .def_readwrite("registers_per_thread",
                     &paddle::platform::DevicePythonNode::registers_per_thread)
      .def_readwrite("blocks_per_sm",
                     &paddle::platform::DevicePythonNode::blocks_per_sm)
      .def_readwrite("warps_per_sm",
                     &paddle::platform::DevicePythonNode::warps_per_sm)
      .def_readwrite("occupancy",
                     &paddle::platform::DevicePythonNode::occupancy)
      .def_readwrite("num_bytes",
                     &paddle::platform::DevicePythonNode::num_bytes)
      .def_readwrite("value", &paddle::platform::DevicePythonNode::value)
      .def("__repr__", [](paddle::platform::DevicePythonNode &event_node) {
        std::stringstream ostr;
        ostr << "DevicePythonNode(name='" << event_node.name << "', type='"
             << paddle::platform::StringTracerEventType(event_node.type)
             << "', start_ns=" << event_node.start_ns
             << ", end_ns=" << event_node.end_ns
             << ", device_id=" << event_node.device_id
             << ", context_id=" << event_node.context_id
             << ", stream_id=" << event_node.stream_id << ")";
        return ostr.str();
      });

  py::class_<paddle::platform::HostPythonNode>(m, "HostPythonNode")
      .def(py::init<>())
      .def_readwrite("name", &paddle::platform::HostPythonNode::name)
      .def_readwrite("type", &paddle::platform::HostPythonNode::type)
      .def_readwrite("start_ns", &paddle::platform::HostPythonNode::start_ns)
      .def_readwrite("end_ns", &paddle::platform::HostPythonNode::end_ns)
      .def_readwrite("process_id",
                     &paddle::platform::HostPythonNode::process_id)
      .def_readwrite("thread_id", &paddle::platform::HostPythonNode::thread_id)
      .def_readwrite("correlation_id",
                     &paddle::platform::HostPythonNode::correlation_id)
      .def_readwrite("input_shapes",
                     &paddle::platform::HostPythonNode::input_shapes)
      .def_readwrite("dtypes", &paddle::platform::HostPythonNode::dtypes)
      .def_readwrite("callstack", &paddle::platform::HostPythonNode::callstack)
      .def_readwrite("attributes",
                     &paddle::platform::HostPythonNode::attributes)
      .def_readwrite("op_id", &paddle::platform::HostPythonNode::op_id)
      .def_readwrite("children_node",
                     &paddle::platform::HostPythonNode::children_node_ptrs)
      .def_readwrite("runtime_node",
                     &paddle::platform::HostPythonNode::runtime_node_ptrs)
      .def_readwrite("device_node",
                     &paddle::platform::HostPythonNode::device_node_ptrs)
      .def_readwrite("mem_node",
                     &paddle::platform::HostPythonNode::mem_node_ptrs)
      .def("__repr__", [](paddle::platform::HostPythonNode &event_node) {
        std::stringstream ostr;
        ostr << "HostPythonNode(name='" << event_node.name << "', type='"
             << paddle::platform::StringTracerEventType(event_node.type)
             << "', start_ns=" << event_node.start_ns
             << ", end_ns=" << event_node.end_ns
             << ", process_id=" << event_node.process_id
             << ", thread_id=" << event_node.thread_id << ")";
        return ostr.str();
      });

  py::class_<paddle::platform::Profiler>(m, "_Profiler")
      .def("create",
           &paddle::platform::Profiler::Create,
           py::return_value_policy::take_ownership)
      .def("is_cupti_supported", &paddle::platform::Profiler::IsCuptiSupported)
      .def("is_cnpapi_supported",
           &paddle::platform::Profiler::IsCnpapiSupported)
      .def("is_xpti_supported", &paddle::platform::Profiler::IsXPTISupported)
      .def("prepare",
           [](paddle::platform::Profiler *profiler) {
             platform::EnableHostEventRecorder();
             profiler->Prepare();
           })
      .def("start", &paddle::platform::Profiler::Start)
      .def(
          "stop",
          [](paddle::platform::Profiler *profiler) {
            platform::DisableHostEventRecorder();
            auto result = profiler->Stop();
            framework::StaticGraphExecutorPerfStatistics(
                result->GetNodeTrees());
            return result;
          },
          py::return_value_policy::automatic_reference);

  py::class_<paddle::platform::ProfilerOptions>(m, "ProfilerOptions")
      .def(py::init<>())
      .def_readwrite("trace_switch",
                     &paddle::platform::ProfilerOptions::trace_switch);

  py::class_<phi::RecordEvent>(m, "_RecordEvent")
      .def(py::init([](std::string name, phi::TracerEventType type) {
        return std::make_unique<phi::RecordEvent>(
            name, type, 1, phi::EventRole::kOrdinary);
      }))
      .def("end", [](phi::RecordEvent *event) { event->End(); });

  py::enum_<paddle::platform::TracerMemEventType>(m, "TracerMemEventType")
#define BIND_ENUM_ITEM(name) .value(#name, phi::TracerMemEventType::name)
      FOR_EACH_TRACER_MEM_EVENT_TYPES(BIND_ENUM_ITEM)
#undef BIND_ENUM_ITEM
          ;  // NOLINT

  py::enum_<paddle::platform::TracerEventType>(m, "TracerEventType")
#define BIND_ENUM_ITEM(name) .value(#name, phi::TracerEventType::name)
      FOR_EACH_TRACER_EVENT_TYPES(BIND_ENUM_ITEM)
#undef BIND_ENUM_ITEM
          ;  // NOLINT

  m.def("tracer_event_type_to_string",
        &paddle::platform::StringTracerEventType);
  m.def("tracer_mem_event_type_to_string",
        &paddle::platform::StringTracerMemEventType);
  m.def("load_profiler_result", &paddle::platform::LoadProfilerResult);
  m.def("enable_memory_recorder", &paddle::platform::EnableMemoryRecorder);
  m.def("disable_memory_recorder", &paddle::platform::DisableMemoryRecorder);
  m.def("enable_op_info_recorder", &phi::EnableOpInfoRecorder);
  m.def("disable_op_info_recorder", &phi::DisableOpInfoRecorder);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  m.def("set_cublas_switch", phi::SetAllowTF32Cublas);
  m.def("get_cublas_switch", phi::AllowTF32Cublas);
  m.def("set_cudnn_switch", phi::SetAllowTF32Cudnn);
  m.def("get_cudnn_switch", phi::AllowTF32Cudnn);
#endif  // PADDLE_WITH_CUDA
  m.def("clear_executor_cache", []() {
    pybind11::gil_scoped_release release;
    framework::InterpreterCoreInfoCache::Instance().Finalize();
  });

  m.def("parse_safe_eager_deletion_skip_vars",
        paddle::framework::details::ParseSafeEagerDeletionSkipVarsSet);

#ifdef PADDLE_WITH_IPU
  py::class_<platform::ipu::IpuBackend,
             std::unique_ptr<platform::ipu::IpuBackend, py::nodelete>>(
      m, "IpuBackend")
      // manage IpuBackend in C++
      .def(
          "get_instance",
          []() {
            return std::unique_ptr<platform::ipu::IpuBackend, py::nodelete>(
                platform::ipu::IpuBackend::GetInstance());
          },
          py::return_value_policy::reference)
      .def("weights_to_host", &platform::ipu::IpuBackend::WeightsToHost)
      .def("detach", &platform::ipu::IpuBackend::Detach)
      .def("reset", &platform::ipu::IpuBackend::Reset)
      .def("set_scope", &platform::ipu::IpuBackend::SetScope)
      .def("set_ipu_strategy", &platform::ipu::IpuBackend::SetIpuStrategy)
      .def("save_model_proto", &platform::ipu::IpuBackend::SaveModelProto);

  py::class_<platform::ipu::IpuStrategy>(m, "IpuStrategy")
      .def(py::init())
      .def("set_options",
           [](platform::ipu::IpuStrategy &self, const py::dict &opt) {
             for (auto element : opt) {
               auto option_name = element.first.cast<std::string>();
               VLOG(10) << "Set option: " << option_name;
               if (option_name == "compilation_progress_logger") {
                 self.SetCompilationProgressLogger(
                     element.second.cast<py::function>());
               } else if (py::isinstance<py::bool_>(element.second)) {
                 self.AddBoolOption(option_name, element.second.cast<bool>());
               } else if (py::isinstance<py::float_>(element.second)) {
                 self.AddDoubleOption(option_name,
                                      element.second.cast<double>());
               } else if (py::isinstance<py::int_>(element.second)) {
                 self.AddUint64Option(option_name,
                                      element.second.cast<std::uint64_t>());
               } else if (py::isinstance<py::str>(element.second)) {
                 self.AddStringOption(option_name,
                                      element.second.cast<std::string>());
               } else if (py::isinstance<py::set>(element.second) ||
                          py::isinstance<py::list>(element.second)) {
                 for (auto option : element.second.cast<py::list>()) {
                   std::string option_val;
                   if (py::isinstance<py::str>(option)) {
                     option_val = option.cast<std::string>();
                   } else if (py::isinstance<py::int_>(option)) {
                     option_val = std::to_string(option.cast<std::uint64_t>());
                   } else {
                     PADDLE_THROW(common::errors::Unimplemented(
                         "Failed to convert type: %s when set IpuStrategy "
                         "option: %s",
                         option.get_type(),
                         option_name));
                   }
                   self.InsertStringOption(option_name, option_val);
                 }
               } else if (py::isinstance<py::dict>(element.second)) {
                 if (option_name.rfind("location_", 0) == 0) {
                   for (auto option : element.second.cast<py::dict>()) {
                     self.SetTensorLocation(
                         option_name,
                         option.first.cast<std::string>(),
                         option.second.cast<std::uint64_t>());
                   }
                 } else if (option_name == "replicated_collectives_settings") {
                   for (auto option : element.second.cast<py::dict>()) {
                     self.SetReplicatedCollectivesSettings(
                         option.first.cast<std::string>(),
                         option.second.cast<bool>());
                   }
                 } else if (option_name == "accumulate_outer_fragment") {
                   for (auto option : element.second.cast<py::dict>()) {
                     std::vector<int> values;
                     for (auto value : option.second.cast<py::list>()) {
                       values.push_back(value.cast<int>());
                     }
                     self.SetAccumulateOuterFragmentSettings(
                         option.first.cast<std::uint64_t>(), values);
                   }
                 } else if (option_name == "custom_op") {
                   std::string paddle_op;
                   std::string popart_op;
                   std::string domain;
                   int version = -1;
                   for (auto option : element.second.cast<py::dict>()) {
                     std::string option_key = option.first.cast<std::string>();
                     if (option_key == "paddle_op") {
                       paddle_op = option.second.cast<std::string>();
                     } else if (option_key == "popart_op") {
                       popart_op = option.second.cast<std::string>();
                     } else if (option_key == "domain") {
                       domain = option.second.cast<std::string>();
                     } else if (option_key == "version") {
                       version = option.second.cast<int>();
                     } else {
                       PADDLE_THROW(common::errors::InvalidArgument(
                           "Invalid argument, key must be one of paddle_op, "
                           "popart_op, domain or version, but revecived %s",
                           option_key));
                     }
                   }
                   self.AddCustomOp(paddle_op, popart_op, domain, version);
                 } else {
                   for (auto option : element.second.cast<py::dict>()) {
                     std::string option_key = option.first.cast<std::string>();
                     std::string option_val;
                     if (py::isinstance<py::str>(option.second)) {
                       option_val = option.second.cast<std::string>();
                     } else if (py::isinstance<py::int_>(option.second)) {
                       option_val =
                           std::to_string(option.second.cast<std::uint64_t>());
                     } else {
                       PADDLE_THROW(common::errors::Unimplemented(
                           "Failed to convert value type: %s when set "
                           "IpuStrategy option: %s",
                           option.second.get_type(),
                           option_key));
                     }
                     self.InsertStringPairOption(
                         option_name, option_key, option_val);
                   }
                 }
               } else {
                 PADDLE_THROW(common::errors::InvalidArgument(
                     "Invalid IpuStrategy option value type: %s, please check "
                     "input value for option: %s",
                     element.second.get_type(),
                     option_name));
               }
             }
           })
      .def("get_option",
           [](platform::ipu::IpuStrategy &self, const std::string &name) {
             py::dict res;
             auto option_type = self.GetOptionType(name);
             res["name"] = name;
             res["type"] = option_type;
             if (option_type == "vector") {
               auto value = self.GetVectorOption(name);
               res["value"] = value;
             } else if (option_type == "map") {
               auto value = self.GetMapOption(name);
               res["value"] = value;
             } else {
               auto value_s = self.GetOption(name);
               res["value_s"] = value_s;
               if (option_type == "bool") {
                 res["value"] = static_cast<bool>(std::stoi(value_s));
               } else if (option_type == "uint64") {
                 res["value"] = std::stoul(value_s);
               } else if (option_type == "double") {
                 res["value"] = std::stod(value_s);
               } else if (option_type == "string") {
                 res["value"] = value_s;
               }
             }
             return res;
           })
      .def("get_all_option_names",
           &platform::ipu::IpuStrategy::GetAllOptionNames)
      .def("enable_pattern", &platform::ipu::IpuStrategy::EnablePattern)
      .def("disable_pattern", &platform::ipu::IpuStrategy::DisablePattern)
      .def("is_pattern_enabled", &platform::ipu::IpuStrategy::IsPatternEnabled);
#endif

  m.def("get_low_precision_op_list", [] {
    py::dict op_list;
    auto list_op = phi::KernelFactory::Instance().GetLowPrecisionKernelList();
    for (auto &op_item : list_op) {
      auto op_name = (op_item.first).c_str();
      auto counts = op_item.second;
      op_list[op_name] = std::to_string(counts.fp16_called_) + "," +
                         std::to_string(counts.bf16_called_) + "," +
                         std::to_string(counts.fp32_called_) + "," +
                         std::to_string(counts.other_called_);
    }
    return op_list;
  });

  m.def("clear_low_precision_op_list",
        [] { phi::KernelFactory::Instance().ClearLowPrecisionKernelList(); });

  m.def("enable_autotune", [] {
    return phi::autotune::AutoTuneStatus::Instance().EnableAutoTune();
  });

  m.def("disable_autotune", [] {
    return phi::autotune::AutoTuneStatus::Instance().DisableAutoTune();
  });

  m.def("set_autotune_range", [](int64_t start, int64_t stop) {
    return phi::autotune::AutoTuneStatus::Instance().SetAutoTuneRange(start,
                                                                      stop);
  });

  m.def("update_autotune_status",
        [] { return phi::autotune::AutoTuneStatus::Instance().Update(); });

  m.def("autotune_status", [] {
    py::dict res;
    phi::autotune::AutoTuneCache::Instance().UpdateStatus();
    res["step_id"] = phi::autotune::AutoTuneStatus::Instance().StepID();
    res["cache_size"] = phi::autotune::AutoTuneCache::Instance().Size();
    res["cache_hit_rate"] =
        phi::autotune::AutoTuneCache::Instance().CacheHitRate();
    return res;
  });

  m.def("enable_layout_autotune",
        [] { return egr::Controller::Instance().EnableLayoutAutoTune(); });

  m.def("disable_layout_autotune",
        [] { return egr::Controller::Instance().DisableLayoutAutoTune(); });

  m.def("use_layout_autotune",
        [] { return egr::Controller::Instance().UseLayoutAutoTune(); });
  // Add the api for nan op debug
  m.def("set_nan_inf_stack_limit",
        &paddle::framework::details::SetNanInfStackLimit);

  // Add the api for nan op debug
  m.def("set_nan_inf_debug_path",
        &paddle::framework::details::SetNanInfDebugPath);

  // Add check op lost
  m.def("set_checked_op_list",
        [](const std::string &op_list) { egr::SetCheckOpList(op_list); });

  // Add skipped op list
  m.def("set_skipped_op_list",
        [](const std::string &op_list) { egr::SetSkipOpList(op_list); });
  BindFleetWrapper(&m);
  BindIO(&m);
  BindCompiledProgram(m);
  BindPlace(m);
  BindTensor(m);

  py::enum_<phi::DataType> data_type(m, "DataType");
  g_data_type_pytype = (PyTypeObject *)data_type.ptr();  // NOLINT
  data_type.value("UNDEFINED", phi::DataType::UNDEFINED)
      .value("BOOL", phi::DataType::BOOL)
      .value("UINT8", phi::DataType::UINT8)
      .value("INT8", phi::DataType::INT8)
      .value("UINT16", phi::DataType::UINT16)
      .value("INT16", phi::DataType::INT16)
      .value("UINT32", phi::DataType::UINT32)
      .value("INT32", phi::DataType::INT32)
      .value("UINT64", phi::DataType::UINT64)
      .value("INT64", phi::DataType::INT64)
      .value("FLOAT32", phi::DataType::FLOAT32)
      .value("FLOAT64", phi::DataType::FLOAT64)
      .value("COMPLEX64", phi::DataType::COMPLEX64)
      .value("COMPLEX128", phi::DataType::COMPLEX128)
      .value("FLOAT16", phi::DataType::FLOAT16)
      .value("BFLOAT16", phi::DataType::BFLOAT16)
      .value("FLOAT8_E4M3FN", phi::DataType::FLOAT8_E4M3FN)
      .value("FLOAT8_E5M2", phi::DataType::FLOAT8_E5M2)
      .value("PSTRING", phi::DataType::PSTRING)
      .value("ALL_DTYPE", phi::DataType::ALL_DTYPE)
      .export_values();

  py::class_<paddle::platform::EngineParams> engine_params(m,
                                                           "TRTEngineParams");
  g_tensorrt_engine_params_pytype =
      reinterpret_cast<PyTypeObject *>(engine_params.ptr());
  engine_params.def(py::init<>())
      .def_readwrite("max_workspace_size",
                     &paddle::platform::EngineParams::max_workspace_size)
      .def_readwrite("min_input_shape",
                     &paddle::platform::EngineParams::min_input_shape)
      .def_readwrite("max_input_shape",
                     &paddle::platform::EngineParams::max_input_shape)
      .def_readwrite("optim_input_shape",
                     &paddle::platform::EngineParams::optim_input_shape)
      .def_readwrite("min_shape_tensor",
                     &paddle::platform::EngineParams::min_shape_tensor)
      .def_readwrite("max_shape_tensor",
                     &paddle::platform::EngineParams::max_shape_tensor)
      .def_readwrite("optim_shape_tensor",
                     &paddle::platform::EngineParams::optim_shape_tensor)
      .def_readwrite("engine_serialized_data",
                     &paddle::platform::EngineParams::engine_serialized_data);

  py::enum_<paddle::framework::ShapeMode>(m, "ShapeMode")
      .value("kMIN", paddle::framework::ShapeMode::kMIN)
      .value("kMAX", paddle::framework::ShapeMode::kMAX)
      .value("kOPT", paddle::framework::ShapeMode::kOPT)
      .export_values();

  m.def("get_value_shape_range_info",
        [](const pir::Value value,
           bool is_shape_tensor,
           paddle::framework::ShapeMode shape_mode) -> py::list {
          py::list res;

          paddle::framework::CollectShapeManager::Instance()
              .StatisticShapeRangeInfo();
          auto shape_result =
              paddle::framework::CollectShapeManager::Instance()
                  .GetValueShapeRangeInfo(value, is_shape_tensor, shape_mode);
          for (auto i : shape_result) {
            res.append(i);
          }
          return res;
        });
  m.def("clear_shape_info", []() {
    paddle::framework::CollectShapeManager::Instance().ClearShapeInfo();
  });
#ifdef PADDLE_WITH_TENSORRT
  m.def("register_paddle_plugin", []() {
    paddle::platform::TrtPluginRegistry::Global()->RegisterToTrt();
  });
#endif

#if defined(PADDLE_WITH_PSLIB) && !defined(PADDLE_WITH_HETERPS)
  BindHeterWrapper(&m);
  BindMetrics(&m);
#endif
#ifdef PADDLE_WITH_HETERPS
  BindPSGPUWrapper(&m);
  BindAfsWrapper(&m);
#endif
  BindGlooWrapper(&m);
  BindBoxHelper(&m);
#ifdef PADDLE_WITH_BOX_PS
  BindBoxWrapper(&m);
#endif
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  BindNCCLWrapper(&m);
#endif
#ifdef PADDLE_WITH_GLOO
  BindGlooContext(&m);
#endif
  BindGraph(&m);
  BindNode(&m);
  BindPass(&m);
  BindInferenceApi(&m);
  BindCompatible(&m);
  BindDataset(&m);
  BindGenerator(&m);
#ifndef PADDLE_NO_PYTHON
  BindDistributed(&m);
#endif
#ifdef PADDLE_WITH_CRYPTO
  BindCrypto(&m);
#endif

#if defined PADDLE_WITH_PSCORE
  BindDistFleetWrapper(&m);
  BindPSHost(&m);
  BindCommunicatorContext(&m);
  BindDistCommunicator(&m);
  BindHeterClient(&m);
  BindGraphPyFeatureNode(&m);
  BindGraphNode(&m);
  BindGraphPyService(&m);
  BindGraphPyServer(&m);
  BindGraphPyClient(&m);
  BindIndexNode(&m);
  BindTreeIndex(&m);
  BindIndexWrapper(&m);
  BindIndexSampler(&m);
#ifdef PADDLE_WITH_HETERPS
  BindNodeQueryResult(&m);
  BindNeighborSampleQuery(&m);
  BindNeighborSampleResult(&m);
  BindGraphGpuWrapper(&m);
#endif
#endif
#if defined(PADDLE_WITH_RPC)
  BindWorkerInfo(&m);
  BindFuture(&m);
  InitAndSetAgentInstance(&m);
  InvokeRpc(&m);
  StartWorker(&m);
  StartClient(&m);
  StopWorker(&m);
  GetWorkerInfo(&m);
  GetWorkerInfoByRank(&m);
  GetCurrentWorkerInfo(&m);
  GetAllWorkerInfos(&m);
#endif

#if defined(PADDLE_WITH_CINN)
  BindTest(&m);
  cinn::pybind::BindCINN(&m);
#endif

  BindPir(&m);
  BindVjp(&m);
  BindDecompRule(&m);
  BindDecompVjp(&m);
#ifdef PADDLE_WITH_DISTRIBUTE
  BindDistApi(&m);
#endif
}
}  // namespace paddle::pybind
