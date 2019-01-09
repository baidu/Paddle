// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <vector>
#include "pybind11/pybind11.h"

#include "Python.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace imperative {

namespace py = ::pybind11;

class PreparedOp {
 public:
  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             framework::OperatorWithKernel::OpKernelFunc func,
             platform::DeviceContext* dev_ctx)
      : op(op), ctx(ctx), func(func), dev_ctx(dev_ctx) {}

  static PreparedOp Prepare(const framework::RuntimeContext& ctx,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(place);

    // check if op[type] has kernel registered.
    auto& all_op_kernels = op.AllOpKernels();
    auto kernels_iter = all_op_kernels.find(op.Type());
    if (kernels_iter == all_op_kernels.end()) {
      PADDLE_THROW(
          "There are no kernels which are registered in the %s operator.",
          op.Type());
    }

    framework::OperatorWithKernel::OpKernelMap& kernels = kernels_iter->second;

    auto expected_kernel_key = op.GetExpectedKernelType(
        framework::ExecutionContext(op, framework::Scope(), *dev_ctx, ctx));
    VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

    auto kernel_iter = kernels.find(expected_kernel_key);
#ifdef PADDLE_WITH_MKLDNN
    // workaround for missing MKLDNN kernel when FLAGS_use_mkldnn env var is set
    if (kernel_iter == kernels.end() &&
        expected_kernel_key.library_type_ == framework::LibraryType::kMKLDNN) {
      VLOG(3) << "missing MKLDNN kernel: fallbacking to PLAIN one";
      expected_kernel_key.library_type_ = framework::LibraryType::kPlain;
      expected_kernel_key.data_layout_ = framework::DataLayout::kAnyLayout;
      kernel_iter = kernels.find(expected_kernel_key);
    }
#endif
    if (kernel_iter == kernels.end()) {
      PADDLE_THROW("op %s does not have kernel for %s", op.Type(),
                   KernelTypeToString(expected_kernel_key));
    }
    return PreparedOp(op, ctx, kernel_iter->second, dev_ctx);
  }

  const framework::OperatorBase& op;
  const framework::RuntimeContext& ctx;
  framework::OperatorWithKernel::OpKernelFunc func;
  platform::DeviceContext* dev_ctx;
};

class OpBase;

class VarBase {
 public:
  VarBase()
      : pre_op_(nullptr),
        pre_op_out_idx_(-1),
        var_desc_(nullptr),
        var_(new framework::Variable()),
        grads_(new framework::Variable()),
        stop_gradient_(false) {}

  explicit VarBase(bool stop_gradient)
      : pre_op_(nullptr),
        pre_op_out_idx_(-1),
        var_desc_(nullptr),
        var_(new framework::Variable()),
        grads_(new framework::Variable()),
        stop_gradient_(stop_gradient) {}

  virtual ~VarBase() {}

  void RunBackward();

  framework::LoDTensor& Grad();

  inline std::string GradName() const {
    PADDLE_ENFORCE(
        var_desc_,
        "Couldn't get gradient variable's name, please call backward() first");
    return string::Sprintf("%s@IGrad", var_desc_->Name());
  }

  OpBase* pre_op_;
  std::string pre_op_out_name_;
  int pre_op_out_idx_;

  framework::VarDesc* var_desc_;
  framework::Variable* var_;
  framework::Variable* grads_;

  bool stop_gradient_;
};

class OpBase {
 public:
  OpBase()
      : op_desc_(nullptr),
        grad_op_desc_(nullptr),
        forward_id_(-1),
        backward_id_(-1) {}

  virtual ~OpBase() {
    if (grad_op_desc_) delete grad_op_desc_;
  }

  std::map<std::string, std::vector<VarBase*>> ApplyGrad();

  framework::OpDesc* op_desc_;
  framework::OpDesc* grad_op_desc_;

  int forward_id_;
  int backward_id_;

  std::map<std::string, std::vector<VarBase*>> input_vars_;
  std::map<std::string, std::vector<VarBase*>> output_vars_;
  std::map<std::string, std::vector<OpBase*>> pre_ops_;
  std::map<std::string, std::vector<int>> pre_ops_out_idx_;

  std::map<std::string, std::vector<framework::Variable*>> grad_input_vars_;
  std::map<std::string, std::vector<framework::Variable*>> grad_output_vars_;
  framework::BlockDesc* block_;
};

class Layer {
 public:
  virtual ~Layer() {}

  virtual std::vector<VarBase> Forward(const std::vector<VarBase>& inputs) {
    std::vector<VarBase> vars;
    return vars;
  }
};

static void CallPythonFunc(const py::object& callable,
                           const std::vector<framework::LoDTensor>& ins,
                           std::vector<VarBase*>* outs) {
  py::gil_scoped_acquire guard;
  py::tuple in_args(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    in_args[i] = ins[i].IsInitialized() ? py::cast(ins[i]) : py::cast(nullptr);
  }

  // TODO(panyx0718): Who owns the returned LoDTensor.
  auto ret = callable(in_args);
  auto ret_tuple = py::cast<py::tuple>(ret);
  size_t ret_num = py::len(ret_tuple);
  for (size_t i = 0; i < ret_num; ++i) {
    try {
      auto* py_out_tensor = py::cast<framework::LoDTensor*>(ret_tuple[i]);
      PADDLE_ENFORCE_NOT_NULL(py_out_tensor,
                              "Output tensor %d should not be nullptr", i);
      VarBase* var = new VarBase();
      auto* tensor = var->var_->GetMutable<framework::LoDTensor>();
      tensor->ShareDataWith(*py_out_tensor);
      tensor->set_lod(py_out_tensor->lod());
      outs->push_back(var);
    } catch (py::cast_error&) {
      PADDLE_THROW("The %d-th output must be LoDTensor", i);
    }
  }
}

static void CallPythonFunc(const py::object& callable,
                           const std::vector<framework::LoDTensor>& ins,
                           std::vector<framework::Variable*>* outs) {
  py::gil_scoped_acquire guard;
  py::tuple in_args(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    in_args[i] = ins[i].IsInitialized() ? py::cast(ins[i]) : py::cast(nullptr);
  }
  VLOG(3) << "pyfunc in " << py::len(in_args);

  // TODO(panyx0718): Who owns the returned LoDTensor.
  auto ret = callable(in_args);
  auto ret_tuple = py::cast<py::tuple>(ret);
  size_t ret_num = py::len(ret_tuple);
  VLOG(3) << "pyfunc out " << ret_num;
  for (size_t i = 0; i < ret_num; ++i) {
    try {
      auto* py_out_tensor = py::cast<framework::LoDTensor*>(ret_tuple[i]);
      PADDLE_ENFORCE_NOT_NULL(py_out_tensor,
                              "Output tensor %d should not be nullptr", i);
      auto* tensor = (*outs)[i]->GetMutable<framework::LoDTensor>();
      tensor->ShareDataWith(*py_out_tensor);
      tensor->set_lod(py_out_tensor->lod());
    } catch (py::cast_error&) {
      PADDLE_THROW("The %d-th output must be LoDTensor", i);
    }
  }
}

class PyLayer {
 public:
  virtual ~PyLayer() {}

  static void RegisterFunc(int func_id, const py::object& py_func);

  static std::vector<VarBase*> Apply(int func_id,
                                     const std::vector<VarBase*>& inputs);

  static void ApplyGrad(int func_id,
                        const std::vector<framework::Variable*>& inputs,
                        std::vector<framework::Variable*>* outputs);
};

}  // namespace imperative
}  // namespace paddle
