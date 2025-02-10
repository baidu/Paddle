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
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/custom_operator/custom_operator_node.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/utils/string/string_helper.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/pybind/cuda_streams_py.h"
#endif

#include "paddle/common/flags.h"
#include "paddle/fluid/eager/custom_operator/custom_operator_utils.h"
#include "paddle/phi/api/include/operants_manager.h"
#include "paddle/phi/api/include/tensor_operants.h"
#include "paddle/phi/api/lib/data_transform.h"
#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#endif

COMMON_DECLARE_string(tensor_operants_mode);

using egr::ConvertAllInputsToDistTensor;
using egr::InputsContainDistTensor;

namespace paddle::pybind {

namespace py = ::pybind11;

extern PyTypeObject* p_tensor_type;
extern PyTypeObject* g_multidevicefeedreader_pytype;
extern PyTypeObject* g_orderedmultidevicefeedreader_pytype;

size_t PyArray_Size_(PyObject* numpy_data) {
  size_t res = 1;
  auto dims = pybind11::detail::array_proxy(numpy_data)->dimensions;
  auto nd = pybind11::detail::array_proxy(numpy_data)->nd;
  while (nd--) {
    res *= (*dims++);
  }
  return res;
}

class EagerNumpyAllocation : public phi::Allocation {
 public:
  explicit EagerNumpyAllocation(PyObject* numpy_data, phi::DataType dtype)
      : Allocation(
            static_cast<void*>(pybind11::detail::array_proxy(numpy_data)->data),
            phi::SizeOf(dtype) * PyArray_Size_(numpy_data),
            phi::CPUPlace()),
        arr_(numpy_data) {
    PADDLE_ENFORCE_NOT_NULL(
        arr_,
        common::errors::InvalidArgument("The underlying PyObject pointer of "
                                        "numpy array cannot be nullptr"));
    PADDLE_ENFORCE_NE(
        arr_,
        Py_None,
        common::errors::PreconditionNotMet(
            "The underlying PyObject pointer of numpy array cannot be None"));
    Py_INCREF(arr_);
  }
  ~EagerNumpyAllocation() override {  // NOLINT
    py::gil_scoped_acquire gil;
    Py_DECREF(arr_);
  }

 private:
  PyObject* arr_;
};

static PyObject* eager_api_scale(PyObject* self,
                                 PyObject* args,
                                 PyObject* kwargs) {
  EAGER_TRY
  // TODO(jiabin): Sync Tensor and Variable here when we support

  auto& tensor =
      reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor;
  float scale = CastPyArg2AttrFloat(PyTuple_GET_ITEM(args, 1), 1);
  float bias = CastPyArg2AttrFloat(PyTuple_GET_ITEM(args, 2), 2);
  bool bias_after_scale = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);
  bool trace_backward = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4);
  paddle::Tensor ret;
  {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    ret = egr::scale(tensor, scale, bias, bias_after_scale, trace_backward);
  }
  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_run_backward(PyObject* self,
                                        PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  auto tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0);
  auto grad_tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 1), 1);
  bool retain_graph = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2);
  const phi::distributed::ProcessMesh* mesh = nullptr;
  if (InputsContainDistTensor(&mesh, tensors, grad_tensors)) {
    tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0, mesh);
    grad_tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 1), 1, mesh);
  }
  {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    egr::Backward(tensors, grad_tensors, retain_graph);
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_run_partial_grad(PyObject* self,
                                            PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  auto tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0);
  auto inputs = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 1), 1);
  auto grad_tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 2), 2);
  auto retain_graph = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);
  auto create_graph = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4);
  auto only_inputs = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 5), 5);
  auto allow_unused = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 6), 6);
  auto no_grad_vars = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 7), 7);
  const phi::distributed::ProcessMesh* mesh = nullptr;
  if (InputsContainDistTensor(
          &mesh, tensors, inputs, grad_tensors, no_grad_vars)) {
    tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0, mesh);
    inputs = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 1), 1, mesh);
    grad_tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 2), 2, mesh);
    no_grad_vars = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 7), 7, mesh);
  }

  std::vector<paddle::Tensor> result;
  {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    result = egr::Grad(tensors,
                       inputs,
                       grad_tensors,
                       retain_graph,
                       create_graph,
                       only_inputs,
                       allow_unused,
                       no_grad_vars);
    VLOG(4) << " in eager_api_run_partial_grad, after running egr::Grad";
  }
  return ToPyObject(result, true /* return_py_none_if_not_initialize */);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_tensor_copy(PyObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor& src =
      reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor;
  paddle::Tensor& dst =
      reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 1))->tensor;
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 2), 2);
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);

  {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    dst = src.copy_to(place, blocking);
    egr::EagerUtils::autograd_meta(&dst)->SetStopGradient(
        egr::EagerUtils::autograd_meta(&(src))->StopGradient());
    egr::EagerUtils::autograd_meta(&dst)->SetPersistable(
        egr::EagerUtils::autograd_meta(&(src))->Persistable());
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* eager_api_get_all_grads(PyObject* self,
                                  PyObject* args,
                                  PyObject* kwargs) {
  EAGER_TRY
  auto tensor_list = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0);

  std::vector<paddle::Tensor> ret;
  for (auto& tensor : tensor_list) {
    VLOG(6) << "Get grad for tensor: " << tensor.name();
    auto meta = egr::EagerUtils::nullable_autograd_meta(tensor);
    if (!meta || meta->StopGradient()) {
      ret.emplace_back(paddle::Tensor());
      continue;
    }
    if (meta && meta->Grad().initialized()) {
      ret.emplace_back(meta->Grad());
    } else {
      ret.emplace_back(paddle::Tensor());
    }
  }
  return ToPyObject(ret, true);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* eager_api_get_grads_lists(PyObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  auto tensor_list = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0);
  // The order of the 3 vectors is: FP16_grads, BF16_grads, FP32_grads
  std::vector<std::vector<paddle::Tensor>> ret(3);

  for (auto& tensor : tensor_list) {
    VLOG(6) << "Get grad for tensor: " << tensor.name();
    auto meta = egr::EagerUtils::nullable_autograd_meta(tensor);
    if (meta && meta->Grad().initialized()) {
      auto& grad = meta->Grad();
      switch (grad.dtype()) {
        case phi::DataType::FLOAT16:
          ret[0].emplace_back(grad);
          break;
        case phi::DataType::BFLOAT16:
          ret[1].emplace_back(grad);
          break;
        case phi::DataType::FLOAT32:
          ret[2].emplace_back(grad);
          break;
        default:
          break;
      }
    }
  }

  return ToPyObject(ret);

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* eager_api_get_grads_types(PyObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  auto tensor_list = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0);

  std::vector<phi::DataType> ret;

  for (auto& tensor : tensor_list) {
    VLOG(6) << "Get grad for tensor: " << tensor.name();
    auto meta = egr::EagerUtils::nullable_autograd_meta(tensor);
    if (!meta || meta->StopGradient()) {
      ret.emplace_back(phi::DataType::UNDEFINED);
      continue;
    }

    auto& grad = meta->Grad();
    if (meta && grad.has_allocation()) {
      if ((grad.is_dense_tensor() || grad.is_dist_tensor()) &&
          (tensor.dtype() == phi::DataType::FLOAT32 ||
           tensor.dtype() == phi::DataType::FLOAT16 ||
           tensor.dtype() == phi::DataType::BFLOAT16)) {
        ret.emplace_back(tensor.dtype());
      }
    } else {
      ret.emplace_back(phi::DataType::UNDEFINED);
    }
  }

  return ToPyObject(ret);

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_read_next_tensor_list(PyObject* self,
                                                 PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  auto tensor_base_list =
      CastPyArg2VectorOfTensorBase(PyTuple_GET_ITEM(args, 0), 0);
  std::vector<paddle::Tensor> tensor_list;
  {
    eager_gil_scoped_release guard;
    tensor_list.reserve(tensor_base_list.size());
    auto func = [](phi::DenseTensor& tensor_base) {
      paddle::Tensor tensor(egr::Controller::Instance().GenerateUniqueName());
      auto autograd_meta = egr::EagerUtils::autograd_meta(&tensor);
      autograd_meta->SetPersistable(false);
      autograd_meta->SetStopGradient(true);
      tensor.set_impl(std::make_shared<phi::DenseTensor>(tensor_base));
      return tensor;
    };
    for (auto& tensor_base : tensor_base_list) {
      tensor_list.emplace_back(func(tensor_base));
    }
  }
  return ToPyObject(tensor_list);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static void ConstructFwdAndBwdMap(
    const std::vector<paddle::OpMetaInfo>& vec_map,
    const std::string& op_type) {
  auto& in_out_map = egr::Controller::Instance().GetCustomEdgesSlotMap();
  if (in_out_map.find(op_type) != in_out_map.end()) {
    VLOG(7) << "Find Exist CustomEdgesSlotMap Skip >>>> ";
    return;
  } else {
    VLOG(7) << "Construct CustomEdgesSlotMap ";
    auto inputs_names = paddle::OpMetaInfoHelper::GetInputs(vec_map[0]);
    auto outputs_names = paddle::OpMetaInfoHelper::GetOutputs(vec_map[0]);
    auto attrs_names = paddle::OpMetaInfoHelper::GetAttrs(vec_map[0]);
    auto grad_outputs_names = paddle::OpMetaInfoHelper::GetOutputs(vec_map[1]);
    auto grad_inputs_names = paddle::OpMetaInfoHelper::GetInputs(vec_map[1]);
    auto grad_attrs_names = paddle::OpMetaInfoHelper::GetAttrs(vec_map[1]);
    std::vector<std::unordered_map<int, int>> res(5);

    in_out_map.insert({op_type, {res}});
    // Prepare pos map for grad_outputs
    VLOG(7) << "Prepare pos map for grad_outputs";
    PADDLE_ENFORCE_LE(
        grad_outputs_names.size(),
        inputs_names.size(),
        common::errors::InvalidArgument(
            "Grad outputs num should be less equal than forward inputs num."));
    for (size_t i = 0; i < grad_outputs_names.size(); i++) {
      size_t end = grad_outputs_names[i].find("@GRAD");
      PADDLE_ENFORCE_NE(
          end,
          std::string::npos,
          common::errors::NotFound(
              "All Grad outputs should be grad and we got %s is not grad var, "
              "please check your op and change to fit the rule.",
              grad_outputs_names[i]));
      for (size_t j = 0; j < inputs_names.size(); j++) {
        if (grad_outputs_names[i].substr(0, end) == inputs_names[j]) {
          VLOG(7) << " ==== Custom Operator: " << op_type << "'s No." << j
                  << " inputs: " << inputs_names[j] << " related to No." << i
                  << " grad_outputs: " << grad_outputs_names[i];
          in_out_map[op_type][0][0][j] = i;  // NOLINT
        }
      }
    }
    // Prepare pos map for grad_inputs
    for (size_t i = 0; i < grad_inputs_names.size(); i++) {
      size_t end = grad_inputs_names[i].find("@GRAD");
      if (end != std::string::npos) {
        for (size_t j = 0; j < outputs_names.size(); j++) {
          if (grad_inputs_names[i].substr(0, end) == outputs_names[j]) {
            VLOG(7) << " ==== Custom Operator: " << op_type << "'s No." << j
                    << " outputs: " << outputs_names[j] << " related to No."
                    << i << " grad_inputs's grad: " << grad_inputs_names[i];
            in_out_map[op_type][0][1][j] = i;  // NOLINT
          }
        }
      } else {
        if (std::find(outputs_names.begin(),
                      outputs_names.end(),
                      grad_inputs_names[i]) != outputs_names.end()) {
          for (size_t j = 0; j < outputs_names.size(); j++) {
            if (grad_inputs_names[i] == outputs_names[j]) {
              VLOG(7) << " ==== Custom Operator: " << op_type << "'s No." << j
                      << " outputs: " << outputs_names[j] << " related to No."
                      << i
                      << " grad_inputs fwd outputs: " << grad_inputs_names[i];
              in_out_map[op_type][0][2][j] = i;  // NOLINT
            }
          }
        } else {
          for (size_t j = 0; j < inputs_names.size(); j++) {
            if (grad_inputs_names[i] == inputs_names[j]) {
              VLOG(7) << " ==== Custom Operator: " << op_type << "'s No." << j
                      << " inputs: " << inputs_names[j] << " related to No."
                      << i
                      << " grad_inputs fwd inputs: " << grad_inputs_names[i];
              in_out_map[op_type][0][3][j] = i;  // NOLINT
            }
          }
        }
      }
    }

    // Prepare pos map for grad attrs_
    for (size_t i = 0; i < grad_attrs_names.size(); i++) {
      auto end = std::find(
          attrs_names.begin(), attrs_names.end(), grad_attrs_names[i]);
      PADDLE_ENFORCE_NE(end,
                        attrs_names.end(),
                        common::errors::NotFound(
                            "All Grad attrs should be one of forward attrs and "
                            "we got %s is not one of them, please check your "
                            "op and change to fit the rule.",
                            grad_attrs_names[i]));
      for (size_t j = 0; j < attrs_names.size(); j++) {
        if (grad_attrs_names[i] == attrs_names[j]) {
          VLOG(7) << " ==== Custom Operator: " << op_type << "'s No." << j
                  << " attrs: " << attrs_names[j] << " related to No." << i
                  << " grad_attrs: " << grad_attrs_names[i];
          in_out_map[op_type][0][4][j] = i;  // NOLINT
        }
      }
    }
  }
}

static PyObject* eager_api_jit_function_call(PyObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY

  std::shared_ptr<jit::Function> function =
      CastPyArg2JitFunction(PyTuple_GET_ITEM(args, 0), 0);
  std::vector<paddle::Tensor> ins =
      CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 1), 1);
  std::vector<paddle::Tensor> outs;
  {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    outs = (*function)(ins);
  }
  return ToPyObject(outs);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api__get_custom_operator_inplace_reverse_idx(
    PyObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  std::string op_type = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 0), 0);
  auto meta_info_map = egr::Controller::Instance().GetOpMetaInfoMap();
  PADDLE_ENFORCE_NE(meta_info_map.find(op_type),
                    meta_info_map.end(),
                    common::errors::NotFound(
                        "Can't find %s in Eager OpMetaInfoMap which should be "
                        "created by LoadOpMetaInfoAndRegisterOp, please make "
                        "sure you registered your op first and try again. ",
                        op_type));

  const auto& inputs =
      paddle::OpMetaInfoHelper::GetInputs(meta_info_map.at(op_type)[0]);
  const auto& outputs =
      paddle::OpMetaInfoHelper::GetOutputs(meta_info_map.at(op_type)[0]);
  const auto& inplace_map =
      paddle::OpMetaInfoHelper::GetInplaceMap(meta_info_map.at(op_type)[0]);
  VLOG(7) << "Custom operator " << op_type
          << " get InplaceMap for python, inplace map size = "
          << inplace_map.size();

  std::unordered_map<int, int> inplace_idx_map;
  for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
    auto& input = inputs[in_idx];
    if (inplace_map.find(input) == inplace_map.end()) {
      continue;
    }
    auto out_iter = find(outputs.begin(), outputs.end(), inplace_map.at(input));
    PADDLE_ENFORCE(out_iter != outputs.end(),
                   common::errors::NotFound(
                       "Can't find the mapped value of %s, please check "
                       "the input of `Inplace` again and make "
                       "sure you registered your op accurately. ",
                       input));
    inplace_idx_map[distance(outputs.begin(), out_iter)] = in_idx;  // NOLINT
  }

  return ToPyObject(inplace_idx_map);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

// This function copies from function `EmptyTensorInitializer` with default
// parameters
static Tensor InitializedEmptyTensor() {
  auto ddims = common::make_ddim({0});
  auto tensor = paddle::Tensor();
  tensor.set_name(
      egr::Controller::Instance().GenerateUniqueName("generated_tensor"));
  auto autograd_meta = egr::EagerUtils::autograd_meta(&tensor);
  autograd_meta->SetPersistable(false);
  std::shared_ptr<phi::DenseTensor> dense_tensor = nullptr;
  std::shared_ptr<phi::Allocation> allocation_ptr = nullptr;
  dense_tensor = std::make_shared<phi::DenseTensor>(
      allocation_ptr, phi::DenseTensorMeta(phi::DataType::FLOAT32, ddims));
  tensor.set_impl(dense_tensor);
  autograd_meta->SetGradNode(
      std::make_shared<egr::GradNodeAccumulation>(autograd_meta));
  return tensor;
}

PyObject* eager_api_run_custom_op(PyObject* self,
                                  PyObject* args,
                                  PyObject* kwargs) {
  EAGER_TRY
  FLAGS_tensor_operants_mode = "phi";
  if (paddle::OperantsManager::Instance().phi_operants.get() == nullptr) {
    paddle::OperantsManager::Instance().phi_operants =
        std::make_unique<paddle::operants::PhiTensorOperants>();
    VLOG(4) << "Initialize phi tensor operants successfully";
  }

  std::string op_type = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 0), 0);
  VLOG(7) << "Get things from python for Custom Op: " << op_type;
  paddle::CustomOpKernelContext ctx;
  auto meta_info_map = egr::Controller::Instance().GetOpMetaInfoMap();
  PADDLE_ENFORCE_NE(meta_info_map.find(op_type),
                    meta_info_map.end(),
                    common::errors::NotFound(
                        "Can't find %s in Eager OpMetaInfoMap which should be "
                        "created by LoadOpMetaInfoAndRegisterOp, please make "
                        "sure you registered your op first and try again. ",
                        op_type));
  const auto& vec_map = meta_info_map.at(op_type);
  const auto& inputs = paddle::OpMetaInfoHelper::GetInputs(vec_map[0]);
  const auto& attrs = paddle::OpMetaInfoHelper::GetAttrs(vec_map[0]);
  const auto& outputs = paddle::OpMetaInfoHelper::GetOutputs(vec_map[0]);
  const auto& inplace_map = paddle::OpMetaInfoHelper::GetInplaceMap(vec_map[0]);

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs.at(i);
    // Parse op_type first, so that use i + 1
    PyObject* obj = PyTuple_GET_ITEM(args, i + 1);
    // Emplace Py_None from python, this means optional inputs passed to C++,
    // use one un-initialized tensor to indicate both Tensor and
    // vector<Tensor> inputs.
    if (obj == Py_None) {
      VLOG(7) << "Custom operator add input " << input
              << " to CustomOpKernelContext. Add un-initialized tensor "
                 "because the optional input is None";
      ctx.EmplaceBackInput(paddle::Tensor());
      continue;
    }
    if (paddle::framework::detail::IsDuplicableVar(input)) {
      std::vector<paddle::Tensor> tensors =
          CastPyArg2VectorOfTensor(obj, i + 1);
      ctx.EmplaceBackInputs(std::move(tensors));
      VLOG(7) << "Custom operator add input " << input
              << " to CustomOpKernelContext. Add vector<Tensor> size = "
              << ctx.InputRangeAt(i).second - ctx.InputRangeAt(i).first;
    } else {
      const paddle::Tensor& tensor = CastPyArg2Tensor(obj, i + 1);  // NOLINT
      ctx.EmplaceBackInput(tensor);
      VLOG(7) << "Custom operator add input " << input
              << " to CustomOpKernelContext. Add Tensor for general case.";
    }
  }

  const phi::distributed::ProcessMesh* mesh = nullptr;
  if (InputsContainDistTensor(&mesh, *(ctx.AllMutableInput()))) {
    paddle::CustomOpKernelContext empty_ctx;
    ctx = empty_ctx;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs.at(i);
      // Parse op_type first, so that use i + 1
      PyObject* obj = PyTuple_GET_ITEM(args, i + 1);
      // Emplace Py_None from python, this means optional inputs passed to C++,
      // use one un-initialized tensor to indicate both Tensor and
      // vector<Tensor> inputs.
      if (obj == Py_None) {
        VLOG(7) << "Custom operator add input " << input
                << " to CustomOpKernelContext. Add un-initialized tensor "
                   "because the optional input is None";
        ctx.EmplaceBackInput(paddle::Tensor());
        continue;
      }
      if (paddle::framework::detail::IsDuplicableVar(input)) {
        std::vector<paddle::Tensor> tensors =
            CastPyArg2VectorOfTensor(obj, i + 1, mesh);
        ctx.EmplaceBackInputs(std::move(tensors));
        VLOG(7) << "Custom operator add input " << input
                << " to CustomOpKernelContext. Add vector<Tensor> size = "
                << ctx.InputRangeAt(i).second - ctx.InputRangeAt(i).first;
      } else {
        paddle::Tensor& tensor = CastPyArg2Tensor(obj, i + 1);  // NOLINT
        ConvertAllInputsToDistTensor(mesh, tensor);
        ctx.EmplaceBackInput(tensor);
        VLOG(7) << "Custom operator add input " << input
                << " to CustomOpKernelContext. Add Tensor for general case.";
      }
    }
  }

  // Parse op_type and inputs first, so that use 1 + inputs.size() + i
  int attr_start_idx = static_cast<int>(1 + inputs.size());
  for (size_t i = 0; i < attrs.size(); ++i) {
    const auto& attr = attrs.at(i);
    std::vector<std::string> attr_name_and_type = paddle::ParseAttrStr(attr);
    auto attr_type_str = attr_name_and_type[1];
    VLOG(7) << "Custom operator add attrs " << attr_name_and_type[0]
            << " to CustomOpKernelContext. Attribute type = " << attr_type_str;
    PyObject* obj = PyTuple_GET_ITEM(args, attr_start_idx + i);
    if (attr_type_str == "bool") {
      ctx.EmplaceBackAttr(
          CastPyArg2AttrBoolean(obj, attr_start_idx + i));  // NOLINT
    } else if (attr_type_str == "int") {
      ctx.EmplaceBackAttr(
          CastPyArg2AttrInt(obj, attr_start_idx + i));  // NOLINT
    } else if (attr_type_str == "float") {
      ctx.EmplaceBackAttr(
          CastPyArg2AttrFloat(obj, attr_start_idx + i));  // NOLINT
    } else if (attr_type_str == "double") {
      ctx.EmplaceBackAttr(
          CastPyArg2AttrDouble(obj, attr_start_idx + i));  // NOLINT
    } else if (attr_type_str == "int64_t") {
      ctx.EmplaceBackAttr(
          CastPyArg2Long(obj, op_type, attr_start_idx + i));  // NOLINT
    } else if (attr_type_str == "std::string") {
      ctx.EmplaceBackAttr(
          CastPyArg2AttrString(obj, attr_start_idx + i));  // NOLINT
    } else if (attr_type_str == "std::vector<int>") {      // NOLINT
      ctx.EmplaceBackAttr(CastPyArg2VectorOfInt(obj, attr_start_idx + i));
    } else if (attr_type_str == "std::vector<float>") {
      ctx.EmplaceBackAttr(CastPyArg2VectorOfFloat(obj, attr_start_idx + i));
    } else if (attr_type_str == "std::vector<int64_t>") {
      ctx.EmplaceBackAttr(
          CastPyArg2Longs(obj, op_type, attr_start_idx + i));  // NOLINT
    } else if (attr_type_str == "std::vector<std::string>") {
      ctx.EmplaceBackAttr(
          CastPyArg2VectorOfString(obj, attr_start_idx + i));  // NOLINT
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported `%s` type value as custom attribute now. "
          "Supported data types include `bool`, `int`, `float`, "
          "`int64_t`, `std::string`, `std::vector<int>`, "
          "`std::vector<float>`, `std::vector<int64_t>`, "
          "`std::vector<std::string>`, Please check whether "
          "the attribute data type and data type string are matched.",
          attr_type_str));
    }
  }

  {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    ctx.ConstructInplaceIndex(inputs, outputs, inplace_map);
    const auto& inplace_reverse_idx_map = ctx.GetInplaceReverseIndexMap();
    for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
      const auto& output = outputs.at(out_idx);
      // inplace special case
      if (inplace_reverse_idx_map.find(out_idx) !=
          inplace_reverse_idx_map.end()) {
        size_t in_idx = inplace_reverse_idx_map.at(out_idx);
        const auto& input_range = ctx.InputRangeAt(in_idx);
        const auto& input_tensor = ctx.InputAt(input_range.first);
        // inplace optional [Tensor or vector<Tensor>], un-initialized tensor.
        if (paddle::framework::detail::IsOptionalVar(output) &&
            !input_tensor.has_allocation()) {
          VLOG(7) << "Custom operator add output " << output
                  << " to CustomOpKernelContext. Add un-initialized tensor "
                     "because the inplace optional input is None";
          ctx.EmplaceBackOutput(paddle::Tensor());
          continue;
        }
        /// inplace vector<Tensor>, initialized tensor.
        if (paddle::framework::detail::IsDuplicableVar(output)) {
          std::vector<paddle::Tensor> empty_tensors;
          size_t vector_size = input_range.second - input_range.first;
          empty_tensors.resize(vector_size);
          for (size_t i = 0; i < vector_size; ++i) {
            empty_tensors[i] = InitializedEmptyTensor();
          }
          VLOG(7) << "Custom operator add output " << output
                  << " to CustomOpKernelContext. Add vector<tensor> size = "
                  << empty_tensors.size();
          ctx.EmplaceBackOutputs(empty_tensors);
          continue;
        }
      }
      VLOG(7) << "Custom operator add output " << output
              << " to CustomOpKernelContext. Add initialized Tensor because "
                 "using general or inplace mechanism";
      // general Tensor or inplace Tensor, initialized tensor.
      ctx.EmplaceBackOutput(InitializedEmptyTensor());
    }

    VLOG(7) << "Run Kernel of Custom Op: " << op_type;
    egr::run_custom_op_impl(vec_map[0], true, false, ctx);

    // handle optional None output when construct backward graph
    for (size_t i = 0; i < ctx.OutputRange().size(); i++) {
      if (ctx.OutputRangeAt(i).first + 1 == ctx.OutputRangeAt(i).second) {
        paddle::Tensor* out_tensor =
            ctx.MutableOutputAt(ctx.OutputRangeAt(i).first);
        if (!out_tensor->initialized()) {
          PADDLE_ENFORCE(
              paddle::framework::detail::IsOptionalVar(outputs.at(i)) ||
                  out_tensor->is_dist_tensor(),
              common::errors::InvalidArgument(
                  "Custom operator[%s]'s %d-th output is not initialized. "
                  "Please check your implementation again. If you are "
                  "using inplace optional output, then you must use "
                  "`paddle::Optional` to decorate this output",
                  op_type,
                  i));
          // We can also consider using `autograd_meta` to tolerant nullptr.
          out_tensor->set_autograd_meta(std::make_shared<egr::AutogradMeta>());
        }
      }
    }

    VLOG(7) << "Get AutogradMeta for inputs and outputs for Custom Op";
    size_t slot_ins_num = ctx.InputRange().size();
    size_t slot_outs_num = ctx.OutputRange().size();
    VLOG(7) << "We got slot num of ins is: " << slot_ins_num;
    VLOG(7) << "We got slot num of outs is: " << slot_outs_num;
    std::vector<egr::AutogradMeta*> ins_auto_grad_metas =
        egr::EagerUtils::nullable_autograd_meta(*ctx.AllMutableInput());
    std::vector<egr::AutogradMeta*> outs_auto_grad_metas =
        egr::EagerUtils::unsafe_autograd_meta(*ctx.AllMutableOutput());

    bool require_any_grad = false;
    bool trace_backward = true;
    for (size_t i = 0; i < ins_auto_grad_metas.size(); ++i) {
      require_any_grad =
          require_any_grad || egr::EagerUtils::ComputeRequireGrad(
                                  trace_backward, ins_auto_grad_metas[i]);
    }

    // handle inplace map
    if (!inplace_map.empty()) {
      for (size_t i = 0; i < ctx.InputRange().size(); i++) {
        if (inplace_map.find(inputs[i]) == inplace_map.end()) {
          continue;
        }
        const auto& input_pair = ctx.InputRangeAt(i);
        for (size_t j = input_pair.first; j < input_pair.second; j++) {
          egr::EagerUtils::CheckInplace(
              ctx.InputAt(j), ins_auto_grad_metas[j], require_any_grad);
          if (ctx.MutableInputAt(j).defined()) {
            // Bump Inplace Version
            ctx.MutableInputAt(j).bump_inplace_version();
            VLOG(3) << "Custom operator: Tensor(" << ctx.InputAt(j).name()
                    << ") uses Inplace Strategy.";
          }
        }
      }
    }

    if (require_any_grad && (vec_map.size() > 1)) {
      VLOG(6) << " Construct Grad for Custom Op: " << op_type;
      ConstructFwdAndBwdMap(vec_map, op_type);
      for (auto& outs_auto_grad_meta : outs_auto_grad_metas) {
        egr::EagerUtils::PassStopGradient(false, outs_auto_grad_meta);
      }
      // Note(HongyuJia): In dygraph eager mode, CheckInplace makes sure leaf
      // nodes set stop_gradient=True. However, dygraph mode can also outputs
      // lead nodes' gradients (For example, we can get x.grad after x.add_(y)).
      // To be consistent with dygraph mode, we have to PassStopGradient for all
      // inplaced ins_auto_grad_metas.
      const auto& inplace_index_map = ctx.GetInplaceIndexMap();
      for (auto pair : inplace_index_map) {
        const auto& size_pair = ctx.InputRangeAt(pair.first);
        for (size_t i = size_pair.first; i < size_pair.second; ++i) {
          egr::EagerUtils::PassStopGradient(false, ins_auto_grad_metas[i]);
        }
      }
      auto grad_node = std::make_shared<egr::RunCustomOpNode>(
          slot_outs_num, slot_ins_num, op_type);
      const auto& slot_map =
          egr::Controller::Instance().GetCustomEdgesSlotMap().at(op_type);

      // Prepare Grad outputs
      size_t no_grad_cnt = 0;
      for (size_t i = 0; i < slot_ins_num; i++) {
        const std::vector<paddle::Tensor>& in_tensors = ctx.InputsBetween(
            ctx.InputRangeAt(i).first, ctx.InputRangeAt(i).second);

        if (slot_map[0][0].find(static_cast<int>(i)) != slot_map[0][0].end()) {
          grad_node->SetGradOutMeta(in_tensors,
                                    slot_map[0][0].at(static_cast<int>(i)));
        } else {
          grad_node->SetGradOutMeta(in_tensors, slot_ins_num - 1 - no_grad_cnt);
          no_grad_cnt++;
        }
      }
      // Prepare Grad inputs with grad of fwd outputs
      for (size_t i = 0; i < slot_outs_num; i++) {
        const auto& size_pair = ctx.OutputRangeAt(i);
        const std::vector<paddle::Tensor>& out_tensors =
            ctx.OutputsBetween(size_pair.first, size_pair.second);
        for (size_t j = size_pair.first; j < size_pair.second; j++) {
          // SetOutRankWithSlot: slot_id = i, rank = j - size_pair.first
          outs_auto_grad_metas[j]->SetSingleOutRankWithSlot(
              i, j - size_pair.first);
          egr::EagerUtils::SetHistory(outs_auto_grad_metas[j], grad_node);
        }
        grad_node->SetGradInMeta(out_tensors, i);
      }

      // Prepare Grad inputs with fwd outputs
      for (auto item : slot_map[0][2]) {
        VLOG(7) << "Prepare fwd_outs: " << item.first
                << " to grad_inputs: " << item.second;
        grad_node->fwd_outs[item.second] =
            egr::RunCustomOpNode::ConstructTensorWrapper(
                ctx.OutputsBetween(ctx.OutputRangeAt(item.first).first,
                                   ctx.OutputRangeAt(item.first).second));
      }

      // Prepare Grad inputs with fwd inputs
      for (auto item : slot_map[0][3]) {
        VLOG(7) << "Prepare fwd_ins: " << item.first
                << " to grad_inputs: " << item.second;
        grad_node->fwd_ins[item.second] =
            egr::RunCustomOpNode::ConstructTensorWrapper(
                ctx.InputsBetween(ctx.InputRangeAt(item.first).first,
                                  ctx.InputRangeAt(item.first).second));
      }

      const std::vector<paddle::any>& res_attrs = ctx.Attrs();
      std::vector<paddle::any> attrs(res_attrs.size());
      // Prepare attrs for Grad node
      for (auto item : slot_map[0][4]) {
        VLOG(7) << "Prepare fwd attrs: " << item.first
                << " to grad_attrs: " << item.second;
        attrs[item.second] = res_attrs[item.first];
      }
      grad_node->SetAttrs(attrs);
    }
  }
  return ToPyObject(*ctx.AllMutableOutput());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_sparse_coo_tensor(PyObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  auto non_zero_indices = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  auto non_zero_elements = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 1), 1);
  auto dense_shape = CastPyArg2VectorOfInt(PyTuple_GET_ITEM(args, 2), 2);
  auto stop_gradient = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);
  paddle::Tensor tensor;
  {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    PADDLE_ENFORCE(
        non_zero_indices.is_dense_tensor(),
        common::errors::Fatal("the non-zero indices must be a DenseTensor."));
    PADDLE_ENFORCE(
        non_zero_elements.is_dense_tensor(),
        common::errors::Fatal("the non-zero elements must be a DenseTensor."));
    auto dense_indices =
        std::dynamic_pointer_cast<phi::DenseTensor>(non_zero_indices.impl());
    auto dense_elements =
        std::dynamic_pointer_cast<phi::DenseTensor>(non_zero_elements.impl());
    // TODO(zhangkaihuo): After creating SparseCooTensor, call coalesced() to
    // sort and merge duplicate indices
    std::shared_ptr<phi::SparseCooTensor> coo_tensor =
        std::make_shared<phi::SparseCooTensor>(
            *dense_indices, *dense_elements, common::make_ddim(dense_shape));
    tensor.set_impl(coo_tensor);
    auto name =
        egr::Controller::Instance().GenerateUniqueName("generated_tensor");
    tensor.set_name(name);
    auto autograd_meta = egr::EagerUtils::autograd_meta(&tensor);
    autograd_meta->SetStopGradient(static_cast<bool>(stop_gradient));
    if (!autograd_meta->GetMutableGradNode()) {
      VLOG(3) << "Tensor(" << name
              << ") doesn't have GradNode, add GradNodeAccumulation to it.";
      autograd_meta->SetGradNode(
          std::make_shared<egr::GradNodeAccumulation>(autograd_meta));
    }
  }
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_sparse_csr_tensor(PyObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  auto non_zero_crows = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  auto non_zero_cols = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 1), 1);
  auto non_zero_elements = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 2), 2);
  auto dense_shape = CastPyArg2VectorOfInt(PyTuple_GET_ITEM(args, 3), 3);
  auto stop_gradient = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4);
  paddle::Tensor tensor;
  {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    PADDLE_ENFORCE(non_zero_crows.is_dense_tensor(),
                   common::errors::Fatal(
                       "the compressed non-zero rows must be a DenseTensor."));
    PADDLE_ENFORCE(
        non_zero_cols.is_dense_tensor(),
        common::errors::Fatal("the non-zero cols must be a DenseTensor."));
    PADDLE_ENFORCE(
        non_zero_elements.is_dense_tensor(),
        common::errors::Fatal("the non-zero elements must be a DenseTensor."));

    auto dense_crows =
        std::dynamic_pointer_cast<phi::DenseTensor>(non_zero_crows.impl());
    auto dense_cols =
        std::dynamic_pointer_cast<phi::DenseTensor>(non_zero_cols.impl());
    auto dense_elements =
        std::dynamic_pointer_cast<phi::DenseTensor>(non_zero_elements.impl());
    std::shared_ptr<phi::SparseCsrTensor> csr_tensor =
        std::make_shared<phi::SparseCsrTensor>(*dense_crows,
                                               *dense_cols,
                                               *dense_elements,
                                               common::make_ddim(dense_shape));
    tensor.set_impl(csr_tensor);
    auto name =
        egr::Controller::Instance().GenerateUniqueName("generated_tensor");
    tensor.set_name(name);
    auto autograd_meta = egr::EagerUtils::autograd_meta(&tensor);
    autograd_meta->SetStopGradient(static_cast<bool>(stop_gradient));
    if (!autograd_meta->GetMutableGradNode()) {
      VLOG(3) << "Tensor(" << name
              << ") have not GradNode, add GradNodeAccumulation for it.";
      autograd_meta->SetGradNode(
          std::make_shared<egr::GradNodeAccumulation>(autograd_meta));
    }
  }
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_register_saved_tensors_hooks(PyObject* self,
                                                        PyObject* args,
                                                        PyObject* kwargs) {
  EAGER_TRY
  if (egr::Controller::Instance().HasGrad()) {
    auto pack_hook = PyTuple_GET_ITEM(args, 0);
    auto unpack_hook = PyTuple_GET_ITEM(args, 1);
    egr::SavedTensorsHooks::GetInstance().SetHooks(
        std::make_shared<PackHook>(pack_hook),
        std::make_shared<UnPackHook>(unpack_hook));
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_reset_saved_tensors_hooks(PyObject* self,
                                                     PyObject* args,
                                                     PyObject* kwargs) {
  EAGER_TRY
  egr::SavedTensorsHooks::GetInstance().ResetHooks();
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

#if defined(PADDLE_WITH_CUDA)
static PyObject* eager_api_async_read(PyObject* self,
                                      PyObject* args,
                                      PyObject* kwargs) {
  EAGER_TRY
  auto& src = GetTensorFromArgs("async_read", "src", args, 0, false);
  auto& dst = GetTensorFromArgs("async_read", "dst", args, 1, false);
  auto& index = GetTensorFromArgs("async_read", "index", args, 2, false);
  auto& buffer = GetTensorFromArgs("async_read", "buffer", args, 3, false);
  auto& offset = GetTensorFromArgs("async_read", "offset", args, 4, false);
  auto& count = GetTensorFromArgs("async_read", "count", args, 5, false);
  const phi::distributed::ProcessMesh* mesh = nullptr;
  if (InputsContainDistTensor(&mesh, src, dst, index, buffer, offset, count)) {
    ConvertAllInputsToDistTensor(mesh, src, dst, index, buffer, offset, count);
  }

  {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    PADDLE_ENFORCE_EQ(
        src.is_gpu_pinned(),
        true,
        common::errors::InvalidArgument("Required `src` device should be "
                                        "CUDAPinnedPlace, but received %d.",
                                        src.place()));
    PADDLE_ENFORCE_EQ(
        dst.is_gpu(),
        true,
        common::errors::InvalidArgument(
            "Required `dst` device should be CUDAPlace, but received %d.",
            dst.place()));
    PADDLE_ENFORCE_EQ(
        index.is_cpu(),
        true,
        common::errors::InvalidArgument(
            "Required `index` device should be CPUPlace, but received %d.",
            index.place()));
    PADDLE_ENFORCE_EQ(buffer.is_gpu_pinned(),
                      true,
                      common::errors::InvalidArgument(
                          "Required `buffer` device should be CUDAPinnedPlace, "
                          "but received %d.",
                          buffer.place()));
    PADDLE_ENFORCE_EQ(
        offset.is_cpu(),
        true,
        common::errors::InvalidArgument(
            "Required `offset` device should be CPUPlace, but received %d.",
            offset.place()));
    PADDLE_ENFORCE_EQ(
        count.is_cpu(),
        true,
        common::errors::InvalidArgument(
            "Required `count` device should be CPUPlace, but received %d.",
            count.place()));

    auto& src_tensor = src;
    auto* dst_tensor = &dst;
    auto& index_tensor = index;
    auto* buffer_tensor = &buffer;
    auto& offset_tensor = offset;
    auto& count_tensor = count;
    auto* dst_data = dst_tensor->mutable_data<float>(dst.place());
    const auto& deviceId = paddle::platform::GetCurrentDeviceId();

    PADDLE_ENFORCE_EQ(src_tensor.dims().size(),
                      dst_tensor->dims().size(),
                      common::errors::InvalidArgument(
                          "`src` and `dst` should have same tensor shape, "
                          "except for the first dimension."));
    PADDLE_ENFORCE_EQ(src_tensor.dims().size(),
                      buffer_tensor->dims().size(),
                      common::errors::InvalidArgument(
                          "`src` and `buffer` should have same tensor shape, "
                          "except for the first dimension."));
    for (int i = 1; i < src_tensor.dims().size(); i++) {
      PADDLE_ENFORCE_EQ(
          src_tensor.dims()[i],
          dst_tensor->dims()[i],
          common::errors::InvalidArgument(
              "`src` and `dst` should have the same tensor shape, "
              "except for the first dimension."));
      PADDLE_ENFORCE_EQ(
          src_tensor.dims()[i],
          buffer_tensor->dims()[i],
          common::errors::InvalidArgument(
              "`src` and `buffer` should have the same tensor shape, "
              "except for the first dimension."));
    }
    PADDLE_ENFORCE_EQ(index_tensor.dims().size(),
                      1,
                      common::errors::InvalidArgument(
                          "`index` tensor should be one-dimensional."));

    auto stream = paddle::platform::get_current_stream(deviceId)->raw_stream();

    int64_t numel = 0;  // total copy length
    int64_t copy_flag = offset_tensor.dims()[0];
    int64_t size = src_tensor.numel() / src_tensor.dims()[0];

    if (copy_flag != 0) {
      PADDLE_ENFORCE_EQ(offset_tensor.dims().size(),
                        1,
                        common::errors::InvalidArgument(
                            "`offset` tensor should be one-dimensional."));
      PADDLE_ENFORCE_EQ(count_tensor.dims().size(),
                        1,
                        common::errors::InvalidArgument(
                            "`count` tensor should be one-dimensional."));
      PADDLE_ENFORCE_EQ(offset_tensor.numel(),
                        count_tensor.numel(),
                        common::errors::InvalidArgument(
                            "`offset` and `count` tensor size mismatch."));
      auto* offset_data = offset_tensor.data<int64_t>();
      auto* count_data = count_tensor.data<int64_t>();
      for (int64_t i = 0; i < count_tensor.numel(); i++) {
        numel += count_data[i];
      }
      PADDLE_ENFORCE_LE(
          numel + index_tensor.numel(),
          buffer_tensor->dims()[0],
          common::errors::InvalidArgument("Buffer tensor size is too small."));
      PADDLE_ENFORCE_LE(
          numel + index_tensor.numel(),
          dst_tensor->dims()[0],
          common::errors::InvalidArgument("Target tensor size is too small."));

      int64_t src_offset, dst_offset = 0, c;
      auto* src_data = src_tensor.data<float>();
      for (int64_t i = 0; i < offset_tensor.numel(); i++) {
        src_offset = offset_data[i], c = count_data[i];
        PADDLE_ENFORCE_LE(
            src_offset + c,
            src_tensor.dims()[0],
            common::errors::InvalidArgument("Invalid offset or count index."));
        PADDLE_ENFORCE_LE(
            dst_offset + c,
            dst_tensor->dims()[0],
            common::errors::InvalidArgument("Invalid offset or count index."));
        cudaMemcpyAsync(dst_data + (dst_offset * size),
                        src_data + (src_offset * size),
                        c * size * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream);
        dst_offset += c;
      }
    } else {
      PADDLE_ENFORCE_LE(
          index_tensor.numel(),
          buffer_tensor->dims()[0],
          common::errors::InvalidArgument("Buffer tensor size is too small."));
    }

    // Select the index data to the buffer
    auto index_select = [](const paddle::Tensor& src_tensor,
                           const paddle::Tensor& index_tensor,
                           paddle::Tensor* buffer_tensor) {
      auto* src_data = src_tensor.data<float>();
      auto* index_data = index_tensor.data<int64_t>();
      auto* buffer_data = buffer_tensor->data<float>();
      const int& slice_size =
          src_tensor.numel() / src_tensor.dims()[0];       // NOLINT
      const int& copy_bytes = slice_size * sizeof(float);  // NOLINT
      int64_t c = 0;
      for (int64_t i = 0; i < index_tensor.numel(); i++) {
        std::memcpy(buffer_data + c * slice_size,
                    src_data + index_data[i] * slice_size,
                    copy_bytes);
        c += 1;
      }
    };
    index_select(src_tensor, index_tensor, buffer_tensor);

    // Copy the data to device memory
    cudaMemcpyAsync(dst_data + (numel * size),
                    buffer_tensor->data<float>(),
                    index_tensor.numel() * size * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream);
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_async_write(PyObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  auto& src = GetTensorFromArgs("async_write", "src", args, 0, false);
  auto& dst = GetTensorFromArgs("async_write", "dst", args, 1, false);
  auto& offset = GetTensorFromArgs("async_write", "offset", args, 2, false);
  auto& count = GetTensorFromArgs("async_write", "count", args, 3, false);
  const phi::distributed::ProcessMesh* mesh = nullptr;
  if (InputsContainDistTensor(&mesh, src, dst, offset, count)) {
    ConvertAllInputsToDistTensor(mesh, src, dst, offset, count);
  }
  {
    eager_gil_scoped_release guard;
    EagerSetDeviceId();
    PADDLE_ENFORCE_EQ(
        src.is_gpu(),
        true,
        common::errors::InvalidArgument(
            "Required `src` device should be CUDAPlace, but received %d. ",
            src.place()));
    PADDLE_ENFORCE_EQ(dst.is_gpu_pinned(),
                      true,
                      common::errors::InvalidArgument(
                          "Required `dst` device should be CUDAPinnedPlace, "
                          "but received %d. ",
                          dst.place()));
    PADDLE_ENFORCE_EQ(
        offset.is_cpu(),
        true,
        common::errors::InvalidArgument("Required `offset` device should "
                                        "be CPUPlace, but received %d. ",
                                        offset.place()));
    PADDLE_ENFORCE_EQ(
        count.is_cpu(),
        true,
        common::errors::InvalidArgument(
            "Required `count` device should be CPUPlace, but received %d. ",
            count.place()));

    // TODO(daisiming): In future, add index as arguments following
    // async_read.
    auto& src_tensor = src;
    auto* dst_tensor = &dst;
    auto& offset_tensor = offset;
    auto& count_tensor = count;
    const auto& deviceId = paddle::platform::GetCurrentDeviceId();

    PADDLE_ENFORCE_EQ(offset_tensor.dims().size(),
                      1,
                      common::errors::InvalidArgument(
                          "`offset` tensor should be one-dimensional."));
    PADDLE_ENFORCE_EQ(count_tensor.dims().size(),
                      1,
                      common::errors::InvalidArgument(
                          "`count` tensor should be one-dimensional."));
    PADDLE_ENFORCE_EQ(offset_tensor.numel(),
                      count_tensor.numel(),
                      common::errors::InvalidArgument(
                          "`offset` and `count` tensor size mismatch."));
    PADDLE_ENFORCE_EQ(src_tensor.dims().size(),
                      dst_tensor->dims().size(),
                      common::errors::InvalidArgument(
                          "`src` and `dst` should have the same tensor shape, "
                          "except for the first dimension."));
    for (int i = 1; i < src_tensor.dims().size(); i++) {
      PADDLE_ENFORCE_EQ(
          src_tensor.dims()[i],
          dst_tensor->dims()[i],
          common::errors::InvalidArgument(
              "`src` and `dst` should have the same tensor shape, "
              "except for the first dimension."));
    }

    auto stream = paddle::platform::get_current_stream(deviceId)->raw_stream();

    int64_t size = src_tensor.numel() / src_tensor.dims()[0];
    auto* src_data = src_tensor.data<float>();
    auto* dst_data = dst_tensor->data<float>();
    const int64_t* offset_data = offset_tensor.data<int64_t>();
    const int64_t* count_data = count_tensor.data<int64_t>();
    int64_t src_offset = 0, dst_offset, c;
    for (int64_t i = 0; i < offset_tensor.numel(); i++) {
      dst_offset = offset_data[i], c = count_data[i];
      PADDLE_ENFORCE_LE(
          src_offset + c,
          src_tensor.dims()[0],
          common::errors::InvalidArgument("Invalid offset or count index"));
      PADDLE_ENFORCE_LE(
          dst_offset + c,
          dst_tensor->dims()[0],
          common::errors::InvalidArgument("Invalid offset or count index"));
      cudaMemcpyAsync(dst_data + (dst_offset * size),
                      src_data + (src_offset * size),
                      c * size * sizeof(float),
                      cudaMemcpyDeviceToHost,
                      stream);
      src_offset += c;
    }
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_to_uva_tensor(PyObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Running in eager_api_to_uva_tensor.";
  auto new_tensor = std::make_shared<paddle::Tensor>(
      egr::Controller::Instance().GenerateUniqueName());
  PyObject* obj = PyTuple_GET_ITEM(args, 0);
  auto array = py::cast<py::array>(py::handle(obj));

  Py_ssize_t args_num = PyTuple_Size(args);
  int64_t device_id = 0;
  if (args_num > 1) {
    PyObject* Py_device_id = PyTuple_GET_ITEM(args, 1);
    if (Py_device_id) {
      device_id = CastPyArg2AttrLong(Py_device_id, 1);
    }
  }

  if (py::isinstance<py::array_t<int32_t>>(array)) {
    SetUVATensorFromPyArray<int32_t>(new_tensor, array, device_id);
  } else if (py::isinstance<py::array_t<int64_t>>(array)) {
    SetUVATensorFromPyArray<int64_t>(new_tensor, array, device_id);
  } else if (py::isinstance<py::array_t<float>>(array)) {
    SetUVATensorFromPyArray<float>(new_tensor, array, device_id);
  } else if (py::isinstance<py::array_t<double>>(array)) {
    SetUVATensorFromPyArray<double>(new_tensor, array, device_id);
  } else if (py::isinstance<py::array_t<int8_t>>(array)) {
    SetUVATensorFromPyArray<int8_t>(new_tensor, array, device_id);
  } else if (py::isinstance<py::array_t<int16_t>>(array)) {
    SetUVATensorFromPyArray<int16_t>(new_tensor, array, device_id);
  } else if (py::isinstance<py::array_t<phi::dtype::float16>>(array)) {
    SetUVATensorFromPyArray<phi::dtype::float16>(new_tensor, array, device_id);
  } else if (py::isinstance<py::array_t<bool>>(array)) {
    SetUVATensorFromPyArray<bool>(new_tensor, array, device_id);
  } else {
    // obj may be any type, obj.cast<py::array>() may be failed,
    // then the array.dtype will be string of unknown meaning.
    PADDLE_THROW(common::errors::InvalidArgument(
        "Input object type error or incompatible array data type. "
        "tensor.set() supports array with bool, float16, float32, "
        "float64, int8, int16, int32, int64,"
        "please check your input or input array data type."));
  }
  return ToPyObject(*(new_tensor.get()));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}
#endif

static PyObject* eager_api__add_backward_final_hook(PyObject* self,
                                                    PyObject* args,
                                                    PyObject* kwargs) {
  EAGER_TRY
  PyObject* hook_func = PyTuple_GET_ITEM(args, 0);
  egr::Controller::Instance().RegisterBackwardFinalHook(
      std::make_shared<PyVoidHook>(hook_func));
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_set_master_grads(PyObject* self,
                                            PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  // tensor_list is a list of model parameters.
  auto tensor_list = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0);
  for (auto& tensor : tensor_list) {
    VLOG(6) << "set master_grad for tensor: " << tensor.name();
    if (!egr::EagerUtils::IsLeafTensor(tensor)) {
      continue;
    }
    paddle::Tensor* grad = egr::EagerUtils::mutable_grad(tensor);
    PADDLE_ENFORCE_NE(
        grad,
        nullptr,
        common::errors::Fatal("Detected nullptr grad"
                              "Please check if you have manually cleared"
                              "the grad inside autograd_meta"));
    if (((*grad).initialized() || (*grad).is_dist_tensor()) &&
        ((*grad).dtype() == phi::DataType::FLOAT16 ||
         (*grad).dtype() == phi::DataType::BFLOAT16)) {
      auto master_grad =
          paddle::experimental::cast(*grad, phi::DataType::FLOAT32);
      grad->set_impl(master_grad.impl());
    }
    VLOG(6) << "finish setting master_grad for tensor: " << tensor.name();
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* eager__is_run_in_backward(PyObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY

  return ToPyObject(egr::Controller::Instance().GetIsInBackward());

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* eager__for_test_check_cuda_error(PyObject* self,
                                           PyObject* args,
                                           PyObject* kwargs) {
  EAGER_TRY
#ifdef PADDLE_WITH_CUDA
  // 1. wait all kernel finish
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

  // 2. get error state
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());

  // 3. check if cuda 700
  size_t bytes = 256;
  char* cuda_mem;
  char* cpu_mem = new char[bytes + 1];

  cudaMalloc(&cuda_mem, bytes + 1);
  cudaMemset(&cuda_mem, 0, bytes + 1);
  cudaMemcpyAsync(cpu_mem, cuda_mem, bytes, cudaMemcpyDeviceToHost);

  cudaFree(cuda_mem);
  delete[] cpu_mem;
#endif
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef variable_functions[] = {  // NOLINT
    // TODO(jiabin): Remove scale when we have final state tests
    {"scale",
     (PyCFunction)(void (*)())eager_api_scale,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_add_backward_final_hook",
     (PyCFunction)(void (*)())eager_api__add_backward_final_hook,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"run_backward",
     (PyCFunction)(void (*)())eager_api_run_backward,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"run_partial_grad",
     (PyCFunction)(void (*)())eager_api_run_partial_grad,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_get_custom_operator_inplace_map",
     (PyCFunction)(void (*)())
         eager_api__get_custom_operator_inplace_reverse_idx,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_run_custom_op",
     (PyCFunction)(void (*)())eager_api_run_custom_op,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"tensor_copy",
     (PyCFunction)(void (*)())eager_api_tensor_copy,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"get_all_grads",
     (PyCFunction)(void (*)())eager_api_get_all_grads,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"get_grads_lists",
     (PyCFunction)(void (*)())eager_api_get_grads_lists,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"get_grads_types",
     (PyCFunction)(void (*)())eager_api_get_grads_types,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"read_next_tensor_list",
     (PyCFunction)(void (*)())eager_api_read_next_tensor_list,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"jit_function_call",
     (PyCFunction)(void (*)())eager_api_jit_function_call,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    /**sparse functions**/
    {"sparse_coo_tensor",
     (PyCFunction)(void (*)())eager_api_sparse_coo_tensor,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"sparse_csr_tensor",
     (PyCFunction)(void (*)())eager_api_sparse_csr_tensor,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"register_saved_tensors_hooks",
     (PyCFunction)(void (*)())eager_api_register_saved_tensors_hooks,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"reset_saved_tensors_hooks",
     (PyCFunction)(void (*)())eager_api_reset_saved_tensors_hooks,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    /**amp functions**/
    {"set_master_grads",
     (PyCFunction)(void (*)())eager_api_set_master_grads,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_run_in_backward",
     (PyCFunction)(void (*)())eager__is_run_in_backward,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_for_test_check_cuda_error",
     (PyCFunction)(void (*)())eager__for_test_check_cuda_error,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
/**sparse functions**/
#if defined(PADDLE_WITH_CUDA)
    {"async_read",
     (PyCFunction)(void (*)())eager_api_async_read,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"async_write",
     (PyCFunction)(void (*)())eager_api_async_write,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"to_uva_tensor",
     (PyCFunction)(void (*)())eager_api_to_uva_tensor,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
#endif
    {nullptr, nullptr, 0, nullptr}};

void BindFunctions(PyObject* module) {
  if (PyModule_AddFunctions(module, variable_functions) < 0) {
    PADDLE_THROW(common::errors::Fatal(
        "Init Paddle error in BindFunctions(PyModule_AddFunctions)."));
    return;
  }
}

}  // namespace paddle::pybind
