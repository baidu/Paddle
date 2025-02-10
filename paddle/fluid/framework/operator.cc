/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/operator.h"

#include <glog/logging.h>

#include <sstream>
#include <string>
#include <unordered_set>

#include "paddle/common/ddim.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/op_call_stack.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "paddle/fluid/framework/unused_var_check.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/isfinite_op.h"
#include "paddle/fluid/operators/ops_extra_info.h"
#include "paddle/fluid/operators/ops_signature/signatures.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/supplement_tracing.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/compat/get_kerneltype_forvar_utils.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/platform/device/device_wrapper.h"
#include "paddle/phi/core/platform/profiler.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "paddle/phi/core/raw_tensor.h"

namespace phi {
class DenseTensor;
}  // namespace phi

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#include "paddle/phi/core/platform/device/xpu/xpu_op_list.h"
#endif

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#include "paddle/phi/core/platform/onednn_op_list.h"
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/platform/device/gpu/gpu_dnn.h"
#endif

COMMON_DECLARE_bool(benchmark);
COMMON_DECLARE_bool(check_nan_inf);
PD_DECLARE_bool(enable_unused_var_check);
COMMON_DECLARE_bool(run_kp_kernel);
PHI_DECLARE_bool(enable_host_event_recorder_hook);

namespace paddle::framework {

std::vector<std::tuple<phi::Place, LibraryType>> kKernelPriority = {
    std::make_tuple(phi::GPUPlace(0), LibraryType::kCUDNN),
    std::make_tuple(phi::GPUPlace(0), LibraryType::kPlain),
    std::make_tuple(phi::CPUPlace(), LibraryType::kMKLDNN),
    std::make_tuple(phi::CPUPlace(), LibraryType::kPlain),
};

static DDim GetDimsDebug(const Scope& scope,
                         const std::string& name,
                         bool get_actual_dim = false) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return DDim({-1});
  }

  if (var->IsType<phi::DenseTensor>()) {
    const phi::DenseTensor& tensor = var->Get<phi::DenseTensor>();
    return tensor.dims();
  } else if (var->IsType<phi::SelectedRows>()) {
    if (get_actual_dim) {
      return var->Get<phi::SelectedRows>().value().dims();
    } else {
      return var->Get<phi::SelectedRows>().GetCompleteDims();
    }
  } else if (var->IsType<Strings>()) {
    return DDim({static_cast<int64_t>(var->Get<Strings>().size())});
  } else if (var->IsType<phi::SparseCooTensor>()) {
    const phi::SparseCooTensor& tensor = var->Get<phi::SparseCooTensor>();
    return tensor.dims();
  } else if (var->IsType<phi::SparseCsrTensor>()) {
    const phi::SparseCsrTensor& tensor = var->Get<phi::SparseCsrTensor>();
    return tensor.dims();
  } else {
    return DDim({-1});
  }
}

static bool VarInited(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) return false;
  return var->IsInitialized();
}

static std::string GetDtype(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return "";
  }

  if (var->IsType<phi::DenseTensor>()) {
    const phi::DenseTensor& tensor = var->Get<phi::DenseTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "";
    }
    return DataTypeToString(framework::TransToProtoVarType(tensor.dtype()));
  } else if (var->IsType<phi::SelectedRows>()) {
    auto tensor = var->Get<phi::SelectedRows>().value();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "uninited";
    } else {
      return DataTypeToString(framework::TransToProtoVarType(tensor.dtype()));
    }
  } else if (var->IsType<Strings>()) {
    return "strings";
  } else if (var->IsType<phi::SparseCooTensor>()) {
    const phi::SparseCooTensor& tensor = var->Get<phi::SparseCooTensor>();
    if (UNLIKELY(!tensor.initialized())) {
      return "";
    }
    return DataTypeToString(framework::TransToProtoVarType(tensor.dtype()));
  } else if (var->IsType<phi::SparseCsrTensor>()) {
    const phi::SparseCsrTensor& tensor = var->Get<phi::SparseCsrTensor>();
    if (UNLIKELY(!tensor.initialized())) {
      return "";
    }
    return DataTypeToString(framework::TransToProtoVarType(tensor.dtype()));
  } else {
    return "";
  }
}

static std::string GetPlace(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return "";
  }
  auto to_string = [](const phi::Place& p) {
    std::stringstream sstream;
    sstream << p;
    return sstream.str();
  };

  if (var->IsType<phi::DenseTensor>()) {
    const phi::DenseTensor& tensor = var->Get<phi::DenseTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "";
    }
    return to_string(tensor.place());
  } else if (var->IsType<phi::SelectedRows>()) {
    auto tensor = var->Get<phi::SelectedRows>().value();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "uninited";
    } else {
      return to_string(tensor.place());
    }
  } else {
    return "";
  }
}

static int GetRowSize(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return -1;
  }

  if (var->IsType<phi::SelectedRows>()) {
    return static_cast<int>(var->Get<phi::SelectedRows>().rows().size());
  }

  return -1;
}

static LegacyLoD GetLoDDebug(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  auto default_lod = LegacyLoD({{}});

  if (var == nullptr) {
    return default_lod;
  }

  if (var->IsType<phi::DenseTensor>()) {
    const phi::DenseTensor& tensor = var->Get<phi::DenseTensor>();
    return tensor.lod();
  } else {
    return default_lod;
  }
}

RuntimeContext::RuntimeContext(const VariableNameMap& innames,
                               const VariableNameMap& outnames,
                               const Scope& scope) {
  for (auto& var_name_item : innames) {
    std::vector<Variable*>& input_vars = inputs[var_name_item.first];
    input_vars.reserve(var_name_item.second.size());
    for (auto& var_name : var_name_item.second) {
      input_vars.push_back(scope.FindVar(var_name));
    }
  }
  for (auto& var_name_item : outnames) {
    std::vector<Variable*>& output_vars = outputs[var_name_item.first];
    output_vars.reserve(var_name_item.second.size());
    for (auto& var_name : var_name_item.second) {
      output_vars.push_back(scope.FindVar(var_name));
    }
  }
}

RuntimeInferShapeContext::RuntimeInferShapeContext(const OperatorBase& op,
                                                   const RuntimeContext& ctx)
    : op_(op), ctx_(ctx) {}

bool RuntimeInferShapeContext::HasInput(const std::string& name) const {
  // has only one input
  const auto& ins = ctx_.inputs;
  auto it = ins.find(name);
  if (it == ins.end()) {
    return false;
  }
  const auto& in = it->second;
  if (in.empty()) return false;
  PADDLE_ENFORCE_EQ(
      in.size(),
      1UL,
      common::errors::InvalidArgument(
          "Input %s should not contain more than one inputs.", name));
  return in[0] != nullptr;
}

bool RuntimeInferShapeContext::HasOutput(const std::string& name) const {
  // has only one output
  const auto& outs = ctx_.outputs;
  auto it = outs.find(name);
  if (it == outs.end()) {
    return false;
  }
  const auto& out = it->second;
  if (out.empty()) {
    return false;
  }
  PADDLE_ENFORCE_EQ(
      out.size(),
      1UL,
      common::errors::InvalidArgument(
          "Output %s should not contain more than one outputs.", name));
  return out[0] != nullptr;
}

bool RuntimeInferShapeContext::HasAttr(const std::string& name) const {
  return op_.HasAttr(name);
}

bool RuntimeInferShapeContext::HasInputs(const std::string& name) const {
  const auto& ins = ctx_.inputs;
  auto it = ins.find(name);
  if (it == ins.end() || it->second.empty()) {
    return false;
  }
  for (auto& input : it->second) {
    if (input == nullptr) {
      return false;
    }
  }
  return true;
}

bool RuntimeInferShapeContext::HasOutputs(const std::string& name,
                                          bool allow_null) const {
  const auto& outs = ctx_.outputs;
  auto it = outs.find(name);
  if (it == outs.end() || it->second.empty()) {
    return false;
  }
  if (!allow_null) {
    for (auto& output : it->second) {
      if (output == nullptr) return false;
    }
  }
  return true;
}

AttrReader RuntimeInferShapeContext::Attrs() const {
  return AttrReader(op_.Attrs(), op_.RuntimeAttrs());
}

std::vector<std::string> RuntimeInferShapeContext::Inputs(
    const std::string& name) const {
  return op_.Inputs(name);
}

std::vector<std::string> RuntimeInferShapeContext::Outputs(
    const std::string& name) const {
  return op_.Outputs(name);
}

std::string RuntimeInferShapeContext::GetInputNameByIdx(size_t idx) const {
  auto& op_proto =
      paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
  PADDLE_ENFORCE_LT(idx,
                    op_proto->inputs().size(),
                    common::errors::OutOfRange(
                        "The index should be less than the size of inputs of "
                        "operator %s, but got index is %d and size is %d",
                        op_.Type(),
                        idx,
                        op_proto->inputs().size()));
  return op_proto->inputs()[static_cast<int>(idx)].name();
}

std::string RuntimeInferShapeContext::GetOutputNameByIdx(size_t idx) const {
  auto& op_proto =
      paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
  PADDLE_ENFORCE_LT(idx,
                    op_proto->outputs().size(),
                    common::errors::OutOfRange(
                        "The index should be less than the size of outputs of "
                        "operator %s, but got index is %d and size is %d",
                        op_.Type(),
                        idx,
                        op_proto->outputs().size()));
  return op_proto->outputs()[static_cast<int>(idx)].name();
}

void RuntimeInferShapeContext::ShareDim(const std::string& in,
                                        const std::string& out,
                                        size_t i,
                                        size_t j) {
  auto in_it = ctx_.inputs.find(in);
  auto out_it = ctx_.outputs.find(out);
  PADDLE_ENFORCE_NE(in_it,
                    ctx_.inputs.end(),
                    common::errors::NotFound("Input %s does not exist.", in));
  PADDLE_ENFORCE_NE(out_it,
                    ctx_.outputs.end(),
                    common::errors::NotFound("Output %s does not exist.", out));
  PADDLE_ENFORCE_LT(i,
                    in_it->second.size(),
                    common::errors::InvalidArgument(
                        "The index of input dimension is out of range, "
                        "excepted index less than %zu, but received %zu.",
                        in_it->second.size(),
                        i));
  PADDLE_ENFORCE_LT(j,
                    out_it->second.size(),
                    common::errors::InvalidArgument(
                        "The index of output dimension is out of range, "
                        "excepted index less than %zu, but received %zu.",
                        out_it->second.size(),
                        j));

  Variable* in_var = in_it->second[i];
  Variable* out_var = out_it->second[j];

  PADDLE_ENFORCE_EQ(
      in_var->Type(),
      out_var->Type(),
      common::errors::InvalidArgument(
          "The type of input (%s) and output (%s) are inconsistent.", in, out));

  if (in_var->IsType<phi::SelectedRows>()) {
    auto& in_sele_rows = in_var->Get<phi::SelectedRows>();
    auto out_sele_rows = out_var->GetMutable<phi::SelectedRows>();
    out_sele_rows->mutable_value()->Resize(in_sele_rows.value().dims());
    out_sele_rows->set_rows(in_sele_rows.rows());
    out_sele_rows->set_height(in_sele_rows.height());
  } else if (in_var->IsType<phi::DenseTensor>()) {
    auto& in_lod_tensor = in_var->Get<phi::DenseTensor>();
    auto* out_lod_tensor = out_var->GetMutable<phi::DenseTensor>();
    out_lod_tensor->Resize(in_lod_tensor.dims());
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Currently, the input type of ShareDim only can be phi::DenseTensor "
        "or SelectedRows."));
  }
}

void RuntimeInferShapeContext::ShareAllLoD(const std::string& in,
                                           const std::string& out) const {
  auto in_it = ctx_.inputs.find(in);
  auto out_it = ctx_.outputs.find(out);
  PADDLE_ENFORCE_NE(in_it,
                    ctx_.inputs.end(),
                    common::errors::NotFound(
                        "Input [%s] found error in Op [%s]", in, op_.Type()));
  PADDLE_ENFORCE_NE(out_it,
                    ctx_.outputs.end(),
                    common::errors::NotFound(
                        "Output [%s] found error in Op [%s]", out, op_.Type()));

  auto& in_var_list = in_it->second;
  auto& out_var_list = out_it->second;

  PADDLE_ENFORCE_EQ(
      in_var_list.size(),
      out_var_list.size(),
      common::errors::PreconditionNotMet(
          "Op [%s]: Input var size should be equal with output var size",
          op_.Type()));

  auto& out_var_names = op_.Outputs(out);

  for (size_t i = 0; i < in_var_list.size(); ++i) {
    if (out_var_names[i] == framework::kEmptyVarName) {
      continue;
    }

    Variable* in_var = in_var_list[i];
    if (!in_var->IsType<phi::DenseTensor>()) return;
    Variable* out_var = out_var_list[i];
    PADDLE_ENFORCE_EQ(
        out_var->IsType<phi::DenseTensor>(),
        true,
        common::errors::PreconditionNotMet(
            "The %d-th output of Output(%s) must be phi::DenseTensor.",
            i,
            out_var_names[i]));
    auto& in_tensor = in_var->Get<phi::DenseTensor>();
    auto* out_tensor = out_var->GetMutable<phi::DenseTensor>();
    out_tensor->set_lod(in_tensor.lod());
#ifdef PADDLE_WITH_DNNL
    if (in_tensor.layout() != DataLayout::ONEDNN)
#endif
      out_tensor->set_layout(in_tensor.layout());
  }
}

void RuntimeInferShapeContext::ShareLoD(const std::string& in,
                                        const std::string& out,
                                        size_t i,
                                        size_t j) const {
  if (can_skip_lod_) {
    return;
  }
  auto in_it = ctx_.inputs.find(in);
  auto out_it = ctx_.outputs.find(out);
  PADDLE_ENFORCE_NE(in_it,
                    ctx_.inputs.end(),
                    common::errors::NotFound("Input %s does not exist.", in));
  PADDLE_ENFORCE_NE(out_it,
                    ctx_.outputs.end(),
                    common::errors::NotFound("Output %s does not exist.", out));
  PADDLE_ENFORCE_LT(i,
                    in_it->second.size(),
                    common::errors::InvalidArgument(
                        "The index of input dimension is out of range, "
                        "excepted index less than %zu, but received %zu.",
                        in_it->second.size(),
                        i));
  PADDLE_ENFORCE_LT(j,
                    out_it->second.size(),
                    common::errors::InvalidArgument(
                        "The index of output dimension is out of range, "
                        "excepted index less than %zu, but received %zu.",
                        out_it->second.size(),
                        j));

  Variable* in_var = in_it->second.at(i);
  if (!in_var->IsType<phi::DenseTensor>()) return;
  Variable* out_var = out_it->second.at(j);
  PADDLE_ENFORCE_EQ(
      out_var->IsType<phi::DenseTensor>(),
      true,
      common::errors::InvalidArgument(
          "The %zu-th output of Output(%s) must be phi::DenseTensor.", j, out));
  auto& in_tensor = in_var->Get<phi::DenseTensor>();
  auto* out_tensor = out_var->GetMutable<phi::DenseTensor>();
  out_tensor->set_lod(in_tensor.lod());

// TODO(dzhwinter) : reuse ShareLoD in most operators.
// Need to call ShareLayout explicitly in sequence related ops.
// Shall we have a better method to shared info between in/out phi::DenseTensor?
#ifdef PADDLE_WITH_DNNL
  // Fix me: ugly workaround below
  // Correct solution:
  //    set_layout() should NOT be called here (i.e. ShareLoD). Instead,
  //    layout of output tensor should be set "manually" in Compute()
  //    of each OPKernel. The reason layout should NOT be shared between
  //    input and output "automatically" (now by InferShape()->ShareLoD())
  //    is that layout transform may occur after InferShape().
  // Workaround:
  //    Skip set_layout() when input layout is kMKLDNN
  //    This is to avoid kMKLDNN is populated wrongly into a non-MKLDNN
  //    OPKernel. In all OneDNN OPkernel, set_layout(kMKLDNN) should be called
  //    in Compute()
  if (in_tensor.layout() != DataLayout::ONEDNN)
#endif
    out_tensor->set_layout(in_tensor.layout());
}

int32_t RuntimeInferShapeContext::GetLoDLevel(const std::string& in,
                                              size_t i) const {
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "GetLoDLevel is only used in compile time. The calculation of "
      "output's actual lod is different among operators so that should be "
      "set in the runtime kernel."));
}

void RuntimeInferShapeContext::SetLoDLevel(const std::string& out,
                                           int32_t lod_level,
                                           size_t j) const {
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "SetLoDLevel is only used in compile time. The calculation of "
      "output's actual lod is different among operators so that should be "
      "set in the runtime kernel."));
}

bool RuntimeInferShapeContext::IsRuntime() const { return true; }

bool RuntimeInferShapeContext::IsRunMKLDNNKernel() const {
  try {
    auto& op_with_kernel = dynamic_cast<const OperatorWithKernel&>(op_);
    return ((op_with_kernel.kernel_type()) &&
            (op_with_kernel.kernel_type()->data_layout_ ==
             phi::DataLayout::ONEDNN));
  } catch (std::bad_cast& exp) {
    return false;
  }
}

// TODO(paddle-dev): Can this be template?
paddle::small_vector<InferShapeVarPtr, phi::kInputSmallVectorSize>
RuntimeInferShapeContext::GetInputVarPtrs(const std::string& name) const {
  const std::vector<Variable*>& vars = InputVars(name);
  paddle::small_vector<InferShapeVarPtr, phi::kInputSmallVectorSize> res;
  res.reserve(vars.size());
  res.insert(res.begin(), vars.begin(), vars.end());
  return res;
}

paddle::small_vector<InferShapeVarPtr, phi::kOutputSmallVectorSize>
RuntimeInferShapeContext::GetOutputVarPtrs(const std::string& name) const {
  const std::vector<Variable*>& vars = OutputVars(name);
  paddle::small_vector<InferShapeVarPtr, phi::kOutputSmallVectorSize> res;
  res.reserve(vars.size());
  res.insert(res.begin(), vars.begin(), vars.end());
  return res;
}

DDim RuntimeInferShapeContext::GetInputDim(const std::string& name) const {
  const std::vector<Variable*>& vars = InputVars(name);
  PADDLE_ENFORCE_EQ(
      vars.size(),
      1UL,
      common::errors::InvalidArgument(
          "Input(%s) should hold one element, but now it holds %zu elements.",
          name,
          vars.size()));
  return this->GetDim(vars[0]);
}

std::vector<DDim> RuntimeInferShapeContext::GetInputsDim(
    const std::string& name) const {
  const std::vector<Variable*>& vars = InputVars(name);
  return GetDims(vars);
}

proto::VarType::Type RuntimeInferShapeContext::GetInputVarType(
    const std::string& name) const {
  return GetVarType(InputVars(name).at(0));
}

std::vector<proto::VarType::Type> RuntimeInferShapeContext::GetInputsVarType(
    const std::string& name) const {
  return GetVarTypes(InputVars(name));
}

std::vector<proto::VarType::Type> RuntimeInferShapeContext::GetOutputsVarType(
    const std::string& name) const {
  return GetVarTypes(OutputVars(name));
}

void RuntimeInferShapeContext::SetOutputDim(const std::string& name,
                                            const DDim& dim) {
  auto& vars = OutputVars(name);
  PADDLE_ENFORCE_EQ(
      vars.size(),
      1UL,
      common::errors::InvalidArgument("Output(%s) should hold one element, "
                                      "but now it holds %zu elements.",
                                      name,
                                      vars.size()));
  SetDim(vars[0], dim);
}

void RuntimeInferShapeContext::SetOutputsDim(const std::string& name,
                                             const std::vector<DDim>& dims) {
  auto& vars = OutputVars(name);
  SetDims(vars, dims);
}

const phi::ArgumentMappingFn*
RuntimeInferShapeContext::GetPhiArgumentMappingFn() const {
  return phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_.Type());
}

const phi::KernelSignature*
RuntimeInferShapeContext::GetPhiDefaultKernelSignature() const {
  return &phi::DefaultKernelSignatureMap::Instance().Get(op_.Type());
}

void RuntimeInferShapeContext::SetSkipLoD(bool skip) { can_skip_lod_ = skip; }

bool RuntimeInferShapeContext::HasRuntimeAttributes() const {
  bool is_runtime = false;
  if (phi::DefaultKernelSignatureMap::Instance().Has(op_.Type())) {
    auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
        GetPhiDefaultKernelSignature()->name);
    if (!phi_kernels.empty()) {
      const auto& args_def = phi_kernels.cbegin()->second.args_def();
      const auto& attr_defs = args_def.attribute_defs();
      for (size_t i = 0; i < attr_defs.size(); ++i) {
        if (attr_defs[i].type_index == phi::AttributeType::SCALAR ||
            attr_defs[i].type_index == phi::AttributeType::INT_ARRAY) {
          is_runtime = true;
          break;
        }
      }
    }
  } else {
    is_runtime = true;
  }
  return is_runtime;
}

std::vector<LegacyLoD> RuntimeInferShapeContext::GetOutputsLod(
    const std::string& out) const {
  auto out_it = ctx_.outputs.find(out);
  auto& out_var_list = out_it->second;

  std::vector<LegacyLoD> ret;
  for (auto* out_var : out_var_list) {
    if (out_var != nullptr) {
      auto* out_tensor = out_var->GetMutable<phi::DenseTensor>();
      ret.push_back(out_tensor->lod());
    }
  }
  return ret;
}

std::vector<DDim> RuntimeInferShapeContext::GetOutputsDim(
    const std::string& name) const {
  const std::vector<Variable*>& vars = OutputVars(name);
  std::vector<Variable*> vars_res;
  for (auto var : vars) {
    if (var != nullptr) {
      vars_res.push_back(var);
    }
  }
  return GetDims(vars_res);
}

DDim RuntimeInferShapeContext::GetDim(Variable* var) const {
  PADDLE_ENFORCE_NOT_NULL(
      var, common::errors::InvalidArgument("Input variable is nullptr."));
  if (var->IsType<phi::DenseTensor>()) {
    return var->Get<phi::DenseTensor>().dims();
  } else if (var->IsType<phi::SelectedRows>()) {
    return var->Get<phi::SelectedRows>().GetCompleteDims();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Only phi::DenseTensor or SelectedRows support 'GetDim', but input "
        "Variable's type is %s.",
        ToTypeName(var->Type())));
  }
}

std::vector<DDim> RuntimeInferShapeContext::GetDims(
    const std::vector<Variable*>& vars) const {
  std::vector<DDim> ret;
  ret.reserve(vars.size());
  std::transform(
      vars.begin(), vars.end(), std::back_inserter(ret), [this](Variable* var) {
        return this->GetDim(var);
      });
  return ret;
}

std::vector<DDim> RuntimeInferShapeContext::GetRepeatedDims(
    const std::string& name) const {
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "GetRepeatedDims method only ban be used in compile time."));
}

void RuntimeInferShapeContext::SetDim(Variable* var, const DDim& dim) {
  if (var->IsType<phi::DenseTensor>()) {
    var->GetMutable<phi::DenseTensor>()->Resize(dim);
  } else if (var->IsType<phi::SelectedRows>()) {
    var->GetMutable<phi::SelectedRows>()->set_height(dim[0]);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Variable type error, expect phi::DenseTensor or SelectedRows, but "
        "received "
        "(%s).",
        ToTypeName(var->Type())));
  }
}

void RuntimeInferShapeContext::SetDims(const std::vector<Variable*>& vars,
                                       const std::vector<DDim>& dims) {
  size_t length = vars.size();
  PADDLE_ENFORCE_EQ(length,
                    dims.size(),
                    common::errors::InvalidArgument(
                        "The number of input variables do not match the "
                        "number of input dimensions, the number of variables "
                        "is %zu, the number of dimensions is %zu.",
                        length,
                        dims.size()));
  for (size_t i = 0; i < length; ++i) {
    if (vars[i] == nullptr) {
      continue;
    }
    SetDim(vars[i], dims[i]);
  }
}

void RuntimeInferShapeContext::SetRepeatedDims(const std::string& name,
                                               const std::vector<DDim>& dims) {
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "SetRepeatedDims method only can be used in compile time."));
}

std::vector<proto::VarType::Type> RuntimeInferShapeContext::GetVarTypes(
    const std::vector<Variable*>& vars) const {
  std::vector<proto::VarType::Type> retv;
  retv.resize(vars.size());
  std::transform(
      vars.begin(),
      vars.end(),
      retv.begin(),
      std::bind(std::mem_fn(&RuntimeInferShapeContext::GetVarType),  // NOLINT
                this,
                std::placeholders::_1));
  return retv;
}

proto::VarType::Type RuntimeInferShapeContext::GetVarType(Variable* var) const {
  return ToVarType(var->Type());
}

const std::vector<Variable*>& RuntimeInferShapeContext::InputVars(
    const std::string& name) const {
  auto it = ctx_.inputs.find(name);
  PADDLE_ENFORCE_NE(
      it,
      ctx_.inputs.end(),
      common::errors::NotFound(
          "Operator (%s) does not have the input (%s).", op_.Type(), name));
  return it->second;
}

const std::vector<Variable*>& RuntimeInferShapeContext::OutputVars(
    const std::string& name) const {
  auto it = ctx_.outputs.find(name);
  PADDLE_ENFORCE_NE(
      it,
      ctx_.outputs.end(),
      common::errors::NotFound(
          "Operator (%s) does not have the outputs (%s).", op_.Type(), name));
  return it->second;
}

void OperatorBase::Run(const Scope& scope, const phi::Place& place) {
  try {
    VLOG(4) << place << " " << DebugStringEx(&scope);
    if (phi::is_gpu_place(place)) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
      PADDLE_THROW(common::errors::Unavailable(
          "Cannot run operator on place %s, please recompile paddle or "
          "reinstall Paddle with CUDA support.",
          place));
#else
      auto dev_id = place.device;
      platform::SetDeviceId(dev_id);
#endif
    } else if (phi::is_xpu_place(place)) {
#ifndef PADDLE_WITH_XPU
      PADDLE_THROW(common::errors::Unavailable(
          "Cannot run operator on place %s, please recompile paddle or "
          "reinstall Paddle with XPU support.",
          place));
#else
      auto dev_id = place.device;
      platform::SetXPUDeviceId(dev_id);
#endif
    } else if (phi::is_custom_place(place)) {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
      PADDLE_THROW(common::errors::Unavailable(
          "Cannot run operator on place %s, please recompile paddle or "
          "reinstall Paddle with CustomDevice support.",
          place));
#else
      phi::DeviceManager::SetDevice(place);
#endif
    }

    {
      // TODO(wangchaochaohu) : refine code to use only one RecordEvent)
      // in order to record different op type cost time
      // and different op name cost time,we set two event.
      phi::RecordEvent op_type_record_event(
          Type(), phi::TracerEventType::Operator, 1);
      auto op_name = platform::OpName(outputs_, Type());
      phi::RecordEvent op_name_record_event(
          op_name,
          phi::TracerEventType::Operator,
          FLAGS_enable_host_event_recorder_hook ? 20 : 1,
          phi::EventRole::kUniqueOp);
      RunImpl(scope, place);
    }

    VLOG(3) << GetExecutionPlace(place) << " " << DebugStringEx(&scope);
  } catch (platform::EnforceNotMet& exception) {
    framework::InsertCallStackInfo(Type(), Attrs(), &exception);
    throw exception;
  } catch (platform::EOFException&) {
    std::rethrow_exception(std::current_exception());
  } catch (std::exception& ex) {
    LOG(WARNING) << Type() << " raises an exception "
                 << common::demangle(typeid(ex).name()) << ", " << ex.what();
    std::rethrow_exception(std::current_exception());
  } catch (...) {
    LOG(WARNING) << Type() << " raises an unknown exception";
    std::rethrow_exception(std::current_exception());
  }
}

bool OperatorBase::HasInputs(const std::string& name) const {
  return inputs_.find(name) != inputs_.end();
}

std::string OperatorBase::Input(const std::string& name) const {
  auto& ins = Inputs(name);
  PADDLE_ENFORCE_LE(
      ins.size(),
      1UL,
      common::errors::InvalidArgument(
          "Operator %s's input %s should contain only one variable.",
          type_,
          name));
  return ins.empty() ? kEmptyVarName : ins[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE_NE(
      it,
      inputs_.end(),
      common::errors::NotFound(
          "Operator %s does not have the input %s.", type_, name));
  return it->second;
}

bool OperatorBase::HasOutputs(const std::string& name) const {
  if (outputs_.find(name) != outputs_.end()) {
    return true;
  } else {
    return false;
  }
}

std::string OperatorBase::Output(const std::string& name) const {
  auto& outs = Outputs(name);
  PADDLE_ENFORCE_LE(
      outs.size(),
      1UL,
      common::errors::InvalidArgument(
          "Operator %s's output %s should contain only one variable.",
          type_,
          name));
  return outs.empty() ? kEmptyVarName : outs[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE_NE(
      it,
      outputs_.end(),
      common::errors::NotFound(
          "Operator %s does not have an output called %s.", type_, name));
  return it->second;
}

std::string OperatorBase::DebugStringEx(const Scope* scope) const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";

  const std::unordered_set<std::string>* no_need_buffer_vars = nullptr;
  if (info_ && info_->NoNeedBufferVarsInferer()) {
    no_need_buffer_vars =
        &(Info().NoNeedBufferVarsInferer()(Inputs(), Outputs(), Attrs()));
    if (no_need_buffer_vars->empty()) no_need_buffer_vars = nullptr;
  }

  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    bool is_no_need_buffer_var =
        (no_need_buffer_vars && no_need_buffer_vars->count(input.first) > 0);
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      auto var_name = input.second[i];
      ss << var_name;
      if (scope) {
        if (!VarInited(*scope, var_name)) {
          ss << "[uninited]";
        } else {
          int row_size = GetRowSize(*scope, var_name);
          if (row_size >= 0) {
            ss << "[row_size=" << row_size << "]";
          }
          std::string dtype = is_no_need_buffer_var
                                  ? "unknown_dtype"
                                  : GetDtype(*scope, var_name);
          std::string place = is_no_need_buffer_var
                                  ? "unknown_place"
                                  : GetPlace(*scope, var_name);
          ss << ":" << dtype;
          ss << "[" << GetDimsDebug(*scope, var_name, true) << "]";
          ss << "(" << GetLoDDebug(*scope, var_name) << ")";
          ss << "(" << place << ")";
        }
      }
      if (i != input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != inputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}, outputs:{";
  for (auto it = outputs_.begin(); it != outputs_.end();) {
    auto& output = *it;
    ss << output.first << "[";
    for (size_t i = 0; i < output.second.size(); ++i) {
      auto var_name = output.second[i];
      ss << var_name;
      if (scope) {
        if (!VarInited(*scope, var_name)) {
          ss << "[uninited]";
        } else {
          int row_size = GetRowSize(*scope, output.second[i]);
          if (row_size >= 0) {
            ss << "[row_size=" << row_size << "]";
          }
          std::string dtype = GetDtype(*scope, output.second[i]);
          ss << ":" << dtype;
          ss << "[" << GetDimsDebug(*scope, var_name, true) << "]";
          ss << "(" << GetLoDDebug(*scope, var_name) << ")";
          ss << "(" << GetPlace(*scope, var_name) << ")";
        }
      }
      if (i != output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != outputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}.";
  return ss.str();
}

OperatorBase::OperatorBase(const std::string& type,
                           const VariableNameMap& inputs,
                           const VariableNameMap& outputs,
                           const AttributeMap& attrs)
    : type_(type),
      inputs_(inputs),
      outputs_(outputs),
      attrs_(attrs),
      // NOTE(zjl): why op_info may be nullptr?
      info_(OpInfoMap::Instance().GetNullable(type)),
      output_hookfuncs_(),
      input_hookfuncs_() {
  // In dygraph mode, all the OperatorBase will be constructed by function:
  // framework::OpRegistry::CreateOp(type, {}, {}, {}, false).
  // Inputs, outputs and attrs will be set to empty map
  // to improve the execution efficiency of dygraph.
  if (!inputs_.empty() || !outputs_.empty()) {
    GenerateTemporaryNames();
    CheckAllInputOutputSet();
  }

  // canonicalize attrs
  if (info_ && info_->proto_) {
    CanonicalizeScalarAttrs(*info_->proto_, &attrs_);
  }
  // In OperatorBase level, all attributes with VarDesc type will be considered
  // as Input.
  for (auto& attr : FilterAttrVar(attrs)) {
    VLOG(3) << "found Attribute with Variable type: " << attr.first;
    inputs_[attr.first] = AttrVarNames(attr.second);
    attrs_.erase(attr.first);
  }
}

std::vector<std::string> OperatorBase::InputVars() const {
  std::vector<std::string> ret_val;
  for (auto& o : inputs_) {
    ret_val.reserve(ret_val.size() + o.second.size());
    ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
  }
  return ret_val;
}

std::vector<std::string> OperatorBase::OutputVars(bool has_intermediate) const {
  std::vector<std::string> ret_val;
  if (has_intermediate) {
    // push all outputs into ret_val
    for (auto& o : outputs_) {
      ret_val.reserve(ret_val.size() + o.second.size());
      ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
    }
    return ret_val;
  }
  auto& info = Info();

  // get all OpProto::Var for outputs
  for (auto& o : info.Proto().outputs()) {
    // ignore all intermediate output
    if (o.intermediate()) continue;
    auto out = outputs_.find(o.name());
    if (out != outputs_.end()) {
      ret_val.reserve(ret_val.size() + out->second.size());
      ret_val.insert(ret_val.end(), out->second.begin(), out->second.end());
    }
  }
  return ret_val;
}

void OperatorBase::CheckAllInputOutputSet() const {
  if (info_ == nullptr || info_->proto_ == nullptr) return;

  for (auto& in : info_->Proto().inputs()) {
    if (!in.dispensable() && !in.extra()) {
      PADDLE_ENFORCE_NE(
          inputs_.find(in.name()),
          inputs_.end(),
          common::errors::NotFound(
              "Operator %s's input (%s) is not set.", Type(), in.name()));
    }
  }

  for (auto& out : info_->Proto().outputs()) {
    if (!out.dispensable() && !out.extra() && !out.intermediate()) {
      PADDLE_ENFORCE_NE(
          outputs_.find(out.name()),
          outputs_.end(),
          common::errors::NotFound(
              "Operator %s's output (%s) is not set.", Type(), out.name()));
    }
  }
}

void OperatorBase::GenerateTemporaryNames() {
  static std::atomic<size_t> gUniqId(0UL);
  for (auto& output : outputs_) {
    for (auto& output_name : output.second) {
      if (output_name == kTempVarName) {
        output_name += type_;
        output_name += "@";
        output_name += std::to_string(gUniqId.fetch_add(1));
      }
    }
  }
}

const phi::DenseTensor* GetDenseTensorOrSelectedRowsValueFromVar(
    const Variable& var) {
  if (var.IsType<phi::DenseTensor>()) {
    return static_cast<const phi::DenseTensor*>(&(var.Get<phi::DenseTensor>()));
  } else if (var.IsType<phi::SelectedRows>()) {
    return &(var.Get<phi::SelectedRows>().value());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Variable type is %s, expect phi::DenseTensor or SelectedRows.",
        ToTypeName(var.Type())));
  }
}

phi::DenseTensor* GetMutableDenseTensorOrSelectedRowsValueFromVar(
    Variable* var) {
  if (var->IsType<phi::DenseTensor>()) {
    return var->GetMutable<phi::DenseTensor>();
  } else if (var->IsType<phi::SelectedRows>()) {
    return var->GetMutable<phi::SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Variable type is %s, expect phi::DenseTensor or SelectedRows.",
        ToTypeName(var->Type())));
  }
}

OperatorWithKernel::OperatorWithKernel(const std::string& type,
                                       const VariableNameMap& inputs,
                                       const VariableNameMap& outputs,
                                       const AttributeMap& attrs)
    : OperatorBase(type, inputs, outputs, attrs), impl_(nullptr) {}

OperatorWithKernel::~OperatorWithKernel() = default;

bool ExecutionContext::HasInput(const std::string& name) const {
  auto* var = InputVar(name);
  return var != nullptr;
}

bool ExecutionContext::HasInputs(const std::string& name) const {
  const auto& ins = ctx_.inputs;
  auto it = ins.find(name);
  if (it == ins.end() || it->second.empty()) {
    return false;
  }
  for (const auto* input : it->second) {
    if (input == nullptr) {
      return false;
    }
  }
  return true;
}

bool ExecutionContext::HasOutput(const std::string& name) const {
  auto* var = OutputVar(name);
  return var != nullptr;
}

const Variable* ExecutionContext::InputVar(const std::string& name) const {
  LogVarUsageIfUnusedVarCheckEnabled(name);

  auto it = ctx_.inputs.find(name);
  if (it == ctx_.inputs.end()) return nullptr;

  PADDLE_ENFORCE_LE(
      it->second.size(),
      1UL,
      common::errors::InvalidArgument(
          "Operator %s's input %s should contain only one variable.",
          op_.Type(),
          name));
  return it->second.empty() ? nullptr : it->second[0];
}

Variable* ExecutionContext::OutputVar(const std::string& name) const {
  auto it = ctx_.outputs.find(name);
  if (it == ctx_.outputs.end()) return nullptr;

  PADDLE_ENFORCE_LE(
      it->second.size(),
      1UL,
      common::errors::InvalidArgument(
          "Operator %s's output %s should contain only one variable.",
          op_.Type(),
          name));
  return it->second.empty() ? nullptr : it->second[0];
}

template <>
const std::vector<const phi::DenseTensor*>
ExecutionContext::MultiInput<phi::DenseTensor>(const std::string& name) const {
  LogVarUsageIfUnusedVarCheckEnabled(name);

  auto vars = MultiInputVar(name);
  if (vars.empty()) {
    return {};
  }
  std::vector<const phi::DenseTensor*> res;
  res.reserve(vars.size());
  std::transform(vars.begin(),
                 vars.end(),
                 std::back_inserter(res),
                 [&](const Variable* var) -> const phi::DenseTensor* {
                   if (var == nullptr) return nullptr;
                   PADDLE_ENFORCE_EQ(
                       var->IsType<phi::DenseTensor>(),
                       true,
                       common::errors::InvalidArgument(
                           "Input variable should be phi::DenseTensor, "
                           "but the received type is %s.",
                           ToTypeName(var->Type())));
                   return &(var->Get<phi::DenseTensor>());
                 });
  return res;
}

template <>
std::vector<phi::DenseTensor*> ExecutionContext::MultiOutput<phi::DenseTensor>(
    const std::string& name) const {
  auto vars = MultiOutputVar(name);

  if (vars.empty()) {
    return {};
  }
  std::vector<phi::DenseTensor*> res;
  res.reserve(vars.size());
  std::transform(vars.begin(),
                 vars.end(),
                 std::back_inserter(res),
                 [&](Variable* var) -> phi::DenseTensor* {
                   return var == nullptr ? nullptr
                                         : var->GetMutable<phi::DenseTensor>();
                 });
  return res;
}

bool OpSupportGPU(const std::string& op_type) {
  // check in new Function kernel first
  bool has_phi_kernel = false;
  auto& kernel_factory = phi::KernelFactory::Instance();
  auto kernel_key_map =
      kernel_factory.SelectKernelMap(phi::TransToPhiKernelName(op_type));
  for (auto& kernel : kernel_key_map) {
    has_phi_kernel = true;
    if (phi::is_gpu_place(phi::TransToPhiPlace(kernel.first.backend()))) {
      return true;
    }
  }

  auto& all_kernels = OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op_type);
  if (it != all_kernels.end()) {
    for (auto& kern_pair : it->second) {
      if (phi::is_gpu_place(kern_pair.first.place_)) {
        return true;
      }
    }
  } else {
    if (has_phi_kernel) {
      // if has phi kernel, but not find phi gpu kernel and fluid gpu kernel,
      // this op doesn't support GPU
      return false;
    } else {
      // All control operator must support GPU
      return true;
    }
  }

  return false;
}

struct OperatorWithKernel::CacheImpl {
  static const char kNotAllowInferShapeCache[];  // NOLINT
  explicit CacheImpl(phi::KernelContext* kernel_ctx,
                     RuntimeInferShapeContext* infer_shape_ctx,
                     const std::vector<phi::DenseTensor*>& tensors,
                     bool not_allow_infer_shape_cache)
      : kernel_ctx_(kernel_ctx),
        infer_shape_ctx_(infer_shape_ctx),
        tensors_(tensors),
        not_allow_infer_shape_cache_(not_allow_infer_shape_cache),
        last_ddims_() {}

  phi::KernelContext* getKernelContext() { return kernel_ctx_.get(); }
  RuntimeInferShapeContext* getRuntimeInferShapeContext() {
    return infer_shape_ctx_.get();
  }

  bool NeedInferShape() {
    if (not_allow_infer_shape_cache_) return true;

    bool ret{false};
    if (last_ddims_.empty() || tensors_.empty()) ret = true;
    if (!ret) {
      PADDLE_ENFORCE_EQ(
          last_ddims_.size(),
          tensors_.size(),
          common::errors::InvalidArgument(
              "The size of last_ddims_ should be equal to tensors_. "));
      for (size_t i = 0; i < last_ddims_.size(); ++i) {
        if (tensors_[i]->dims() != last_ddims_[i]) {
          ret = true;
          break;
        }
      }
    }
    if (ret) {
      last_ddims_.resize(tensors_.size());
      for (size_t i = 0; i < last_ddims_.size(); ++i) {
        last_ddims_[i] = tensors_[i]->dims();
      }
    }
    VLOG(3) << "need infer shape is " << ret;
    return ret;
  }

 private:
  std::unique_ptr<phi::KernelContext> kernel_ctx_;
  std::unique_ptr<RuntimeInferShapeContext> infer_shape_ctx_;
  std::vector<phi::DenseTensor*> tensors_;
  bool not_allow_infer_shape_cache_;
  std::vector<phi::DDim> last_ddims_;
};
const char  // NOLINT
    OperatorWithKernel::CacheImpl::kNotAllowInferShapeCache[] =
        "@NOT_ALLOW_INFERSHAPE_CACHE@";

static void CheckTensorNANOrInf(const std::string& op_type,
                                const std::string& name,
                                const phi::DenseTensor& tensor) {
  if (tensor.memory_size() == 0) {
    return;
  }
  if (framework::TransToProtoVarType(tensor.dtype()) != proto::VarType::FP32 &&
      framework::TransToProtoVarType(tensor.dtype()) != proto::VarType::FP64) {
    return;
  }
  PADDLE_ENFORCE_NE(framework::TensorContainsInf(tensor),
                    true,
                    common::errors::Fatal(
                        "Operator %s output phi::DenseTensor %s contains Inf.",
                        op_type,
                        name));
  PADDLE_ENFORCE_NE(framework::TensorContainsNAN(tensor),
                    true,
                    common::errors::Fatal(
                        "Operator %s output phi::DenseTensor %s contains NAN.",
                        op_type,
                        name));
}

bool OperatorWithKernel::SupportGPU() const {
  auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
      phi::TransToPhiKernelName(type_));
  auto has_phi_kernel =
      std::any_of(phi_kernels.begin(),
                  phi_kernels.end(),
                  [](phi::KernelKeyMap::const_reference kern_pair) {
                    return kern_pair.first.backend() == phi::Backend::GPU;
                  });
  if (has_phi_kernel) {
    return true;
  } else {
    auto kernel_iter = OperatorWithKernel::AllOpKernels().find(type_);
    if (kernel_iter == OperatorWithKernel::AllOpKernels().end()) {
      return false;
    } else {
      auto& op_kernels = kernel_iter->second;
      return std::any_of(op_kernels.begin(),
                         op_kernels.end(),
                         [](OpKernelMap::const_reference kern_pair) {
                           return phi::is_gpu_place(kern_pair.first.place_);
                         });
    }
  }
}

bool OperatorWithKernel::SupportXPU() const {
#ifdef PADDLE_WITH_XPU
  auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
      phi::TransToPhiKernelName(type_));
  auto has_phi_kernel =
      std::any_of(phi_kernels.begin(),
                  phi_kernels.end(),
                  [](phi::KernelKeyMap::const_reference kern_pair) {
                    return kern_pair.first.backend() == phi::Backend::XPU;
                  });
  if (has_phi_kernel) {
    return true;
  } else {
    auto kernel_iter = OperatorWithKernel::AllOpKernels().find(type_);
    if (kernel_iter == OperatorWithKernel::AllOpKernels().end()) {
      return false;
    } else {
      auto& op_kernels = kernel_iter->second;
      return std::any_of(
          op_kernels.begin(),
          op_kernels.end(),
          [this](OpKernelMap::const_reference kern_pair) {
            bool is_xpu_support1 = phi::backends::xpu::is_xpu_support_op(
                type_, phi::TransToPhiDataType(kern_pair.first.data_type_));
            bool is_xpu_support2 = phi::backends::xpu::is_xpu_support_op(
                phi::TransToPhiKernelName(type_),
                phi::TransToPhiDataType(kern_pair.first.data_type_));
            return phi::is_xpu_place(kern_pair.first.place_) &&
                   (is_xpu_support1 || is_xpu_support2);
          });
    }
  }
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "should not call OperatorWithKernel::SupportXPU() when not compiled with "
      "XPU support."));
  return false;
#endif
}

bool OperatorWithKernel::SupportCustomDevice() const {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
      phi::TransToPhiKernelName(type_));
  auto has_phi_kernel =
      std::any_of(phi_kernels.begin(),
                  phi_kernels.end(),
                  [](phi::KernelKeyMap::const_reference kern_pair) {
                    return phi::is_custom_place(
                        phi::TransToPhiPlace(kern_pair.first.backend()));
                  });
  if (has_phi_kernel) {
    return true;
  } else {
    auto kernel_iter = OperatorWithKernel::AllOpKernels().find(type_);
    if (kernel_iter == OperatorWithKernel::AllOpKernels().end()) {
      return false;
    } else {
      auto& op_kernels = kernel_iter->second;
      return std::any_of(op_kernels.begin(),
                         op_kernels.end(),
                         [this](OpKernelMap::const_reference kern_pair) {
                           return phi::is_custom_place(kern_pair.first.place_);
                         });
    }
  }
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "should not call OperatorWithKernel::SupportCustomDevice() when not "
      "compiled with "
      "CustomDevice support."));
  return false;
#endif
}

bool OperatorWithKernel::SupportsMKLDNN(const phi::DataType data_type) const {
  auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
      phi::TransToPhiKernelName(type_));
  auto has_phi_kernel =
      std::any_of(phi_kernels.begin(),
                  phi_kernels.end(),
                  [data_type](phi::KernelKeyMap::const_reference kern_pair) {
                    return kern_pair.first.backend() == phi::Backend::ONEDNN &&
                           kern_pair.first.dtype() == data_type;
                  });
  if (has_phi_kernel) {
    return true;
  } else {
    auto op_kernel_iter = OperatorWithKernel::AllOpKernels().find(type_);
    if (op_kernel_iter == OperatorWithKernel::AllOpKernels().end()) {
      return false;
    } else {
      auto& op_kernels = op_kernel_iter->second;
      return std::any_of(
          op_kernels.begin(),
          op_kernels.end(),
          [data_type](OpKernelMap::const_reference kern_pair) {
            return phi::is_cpu_place(kern_pair.first.place_) &&
                   kern_pair.first.library_type_ == LibraryType::kMKLDNN &&
                   kern_pair.first.data_type_ ==
                       paddle::framework::TransToProtoVarType(data_type);
          });
    }
  }
}

bool OperatorWithKernel::SupportsCUDNN(const phi::DataType data_type) const {
  auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
      phi::TransToPhiKernelName(type_));
  auto has_phi_kernel =
      std::any_of(phi_kernels.begin(),
                  phi_kernels.end(),
                  [data_type](phi::KernelKeyMap::const_reference kern_pair) {
                    return kern_pair.first.backend() == phi::Backend::GPUDNN &&
                           kern_pair.first.dtype() == data_type;
                  });
  if (has_phi_kernel) {
    return true;
  } else {
    auto op_kernel_iter = OperatorWithKernel::AllOpKernels().find(type_);
    if (op_kernel_iter == OperatorWithKernel::AllOpKernels().end()) {
      return false;
    } else {
      auto& op_kernels = op_kernel_iter->second;
      proto::VarType::Type fluid_data_type =
          framework::TransToProtoVarType(data_type);
      return std::any_of(
          op_kernels.begin(),
          op_kernels.end(),
          [fluid_data_type](OpKernelMap::const_reference kern_pair) {
            return phi::is_gpu_place(kern_pair.first.place_) &&
                   kern_pair.first.library_type_ == LibraryType::kCUDNN &&
                   kern_pair.first.data_type_ == fluid_data_type;
          });
    }
  }
}

bool OperatorWithKernel::SupportsCPUBF16() const {
  auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
      phi::TransToPhiKernelName(type_));
  auto has_phi_kernel =
      std::any_of(phi_kernels.begin(),
                  phi_kernels.end(),
                  [](phi::KernelKeyMap::const_reference kern_pair) {
                    return kern_pair.first.backend() == phi::Backend::CPU &&
                           kern_pair.first.dtype() == phi::DataType::BFLOAT16;
                  });
  if (has_phi_kernel) {
    return true;
  } else {
    auto op_kernel_iter = OperatorWithKernel::AllOpKernels().find(type_);
    if (op_kernel_iter == OperatorWithKernel::AllOpKernels().end()) {
      return false;
    } else {
      auto& op_kernels = op_kernel_iter->second;
      return std::any_of(op_kernels.begin(),
                         op_kernels.end(),
                         [](OpKernelMap::const_reference kern_pair) {
                           return phi::is_cpu_place(kern_pair.first.place_) &&
                                  kern_pair.first.place_ == phi::CPUPlace() &&
                                  kern_pair.first.data_type_ ==
                                      proto::VarType::Type::VarType_Type_BF16;
                         });
    }
  }
}

bool OperatorWithKernel::SupportsKernelType(
    const OpKernelType& kernel_type, const ExecutionContext& exe_ctx) const {
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  if (kernels_iter == all_op_kernels.end()) return false;
  OpKernelMap& kernels = kernels_iter->second;
  auto kernel_iter = kernels.find(kernel_type);

#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
  if (phi::is_xpu_place(kernel_type.place_)) {
    return kernel_iter != kernels.end() &&
           paddle::platform::is_xpu_support_op(
               type_, phi::TransToPhiDataType(kernel_type.data_type_));
  }
#endif

#ifdef PADDLE_WITH_XPU_KP
  if (phi::is_xpu_place(kernel_type.place_)) {
    bool use_xpu_kp_kernel_rt =
        FLAGS_run_kp_kernel &&
        paddle::platform::is_xpu_kp_support_op(
            type_, phi::TransToPhiDataType(kernel_type.data_type_));
    bool use_xpu_kp_kernel_debug =
        paddle::platform::is_in_xpu_kpwhite_list(type_);
    bool is_xpu_kp_support = (use_xpu_kp_kernel_rt || use_xpu_kp_kernel_debug);
    if (is_xpu_kp_support) {
      auto tmp_kernel_type = kernel_type;
      tmp_kernel_type.library_type_ = LibraryType::kKP;
      return kernels.find(tmp_kernel_type) != kernels.end();
    }
    return kernel_iter != kernels.end() &&
           paddle::platform::is_xpu_support_op(
               type_, phi::TransToPhiDataType(kernel_type.data_type_));
  }
#endif

// NOTE(jiahongyu): If OneDNN can be used, the function SupportsKernelType needs
// to check whether current op supports OneDNN kernel. There are three
// statements in if condition:
// 1. Whether onednn kernel fallbacks to plain kernel;
// 2. Whether this op has specific implementation;
// 3. Whether onednn kernel can be used.
#ifdef PADDLE_WITH_DNNL
  if (!this->DnnFallback() && !paddle::platform::in_onednn_white_list(type_) &&
      this->CanMKLDNNBeUsed(exe_ctx, kernel_type.data_type_)) {
    auto tmp_kernel_type = kernel_type;
    tmp_kernel_type.library_type_ = framework::LibraryType::kMKLDNN;
    tmp_kernel_type.data_layout_ = framework::DataLayout::ONEDNN;
    return kernels.find(tmp_kernel_type) != kernels.end();
  }
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (this->CanCUDNNBeUsed(exe_ctx, kernel_type.data_type_)) {
    auto tmp_kernel_type = kernel_type;
    tmp_kernel_type.library_type_ = framework::LibraryType::kCUDNN;
    return kernels.find(tmp_kernel_type) != kernels.end();
  }
#endif

  return kernel_iter != kernels.end();
}

bool OperatorWithKernel::CanMKLDNNBeUsed(const framework::ExecutionContext& ctx,
                                         phi::DataType data_type) const {
  return ctx.HasAttr("use_mkldnn") && ctx.Attr<bool>("use_mkldnn") &&
         phi::is_cpu_place(ctx.GetPlace()) && this->SupportsMKLDNN(data_type);
}

bool OperatorWithKernel::CanMKLDNNBeUsed(const framework::ExecutionContext& ctx,
                                         proto::VarType::Type data_type) const {
  return this->CanMKLDNNBeUsed(ctx, phi::TransToPhiDataType(data_type));
}

bool OperatorWithKernel::CanCUDNNBeUsed(const framework::ExecutionContext& ctx,
                                        phi::DataType data_type) const {
  bool use_cudnn = ctx.HasAttr("use_cudnn") && ctx.Attr<bool>("use_cudnn") &&
                   phi::is_gpu_place(ctx.GetPlace());

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (use_cudnn) {
    auto& dev_ctx = ctx.device_context<phi::GPUContext>();
    use_cudnn &= (dev_ctx.cudnn_handle() != nullptr);
  }
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

#if defined(PADDLE_WITH_CUDA)
  if (use_cudnn && data_type == phi::DataType::BFLOAT16) {
    PADDLE_ENFORCE_GE(
        platform::DnnVersion(),
        8100,
        common::errors::InvalidArgument(
            "bfloat16 can only be used when CUDNN_VERSION >= 8100"));
  }
#endif  // PADDLE_WITH_CUDA

  return use_cudnn && this->SupportsCUDNN(data_type);
}

bool OperatorWithKernel::CanCUDNNBeUsed(const framework::ExecutionContext& ctx,
                                        proto::VarType::Type data_type) const {
  return this->CanCUDNNBeUsed(ctx, phi::TransToPhiDataType(data_type));
}

void OperatorWithKernel::InferShape(InferShapeContext* ctx) const {
  PADDLE_THROW(common::errors::PermissionDenied(
      "The default InferShape function of OperatorWithKernel is not allowed to "
      "be called, please override corresponding InferShape function in the "
      "specific operator."));
}

void OperatorWithKernel::RuntimeInferShape(const Scope& scope,
                                           const phi::Place& place,
                                           const RuntimeContext& ctx) const {
  RuntimeInferShapeContext infer_shape_ctx(*this, ctx);
  this->Info().infer_shape_(&infer_shape_ctx);
}

template <typename T>
bool HasSameTensorType(phi::TensorBase* phi_tensor, Variable* var) {
  if (phi_tensor == nullptr && var == nullptr) {
    return true;
  } else if (phi_tensor != nullptr && var != nullptr) {
    if (T::classof(phi_tensor) && var->IsType<T>()) {
      return true;
    }
  }
  return false;
}

// TODO(YuanRisheng): We need collect all `need_prepare_phi_data_`
// into this function.
void OperatorWithKernel::CheckWhetherPreparePhiData(
    const VariableNameMap& innames,
    const VariableNameMap& outnames,
    const Scope& scope) const {
  if (run_phi_kernel_ && impl_ != nullptr) {
    const auto& phi_kernel_context = impl_->getKernelContext();
    size_t phi_tensor_index = 0;
    // Check each tensor in KernelContext, if there is a tensor that has
    // different type with variable. The PhiKernelContext need be reconstructed.
    // We use kernel_signature_'s output to retrieve tensor. Because the tensor
    // in phi_kernel_context stored in the order of kernel_signature_'s output.
    if (phi_kernel_context->OutputsSize() >= phi_tensor_index ||
        kernel_signature_ == nullptr) {
      need_prepare_phi_data_ = true;
      return;
    }

    const auto& phi_output_names = kernel_signature_->output_names;
    for (auto& phi_output_name : phi_output_names) {
      const auto& iter = outnames.find(phi_output_name);
      if (iter != outnames.end()) {
        for (auto& var_name : iter->second) {
          auto var_output = scope.FindVar(var_name);
          auto phi_output =
              phi_kernel_context->MutableOutputAt<phi::TensorBase>(
                  phi_tensor_index);
          if (phi_output == nullptr) {
            continue;
          }
          if (!(HasSameTensorType<phi::DenseTensor>(phi_output, var_output) ||
                HasSameTensorType<phi::SparseCooTensor>(phi_output,
                                                        var_output) ||
                HasSameTensorType<phi::Strings>(phi_output, var_output))) {
            need_prepare_phi_data_ = true;
          }
          phi_tensor_index++;
        }
      }
    }
  }
}

void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const phi::Place& place) const {
  // To reduce the elapsed time of HasAttr, we use bool variable to record the
  // result of HasAttr.
  if (!enable_cache_runtime_context_ && HasAttr(kEnableCacheRuntimeContext))
    enable_cache_runtime_context_ = true;
  if (!all_kernels_must_compute_runtime_shape_ &&
      HasAttr(kAllKernelsMustComputeRuntimeShape))
    all_kernels_must_compute_runtime_shape_ = true;
  const Scope* cur_scope = &scope;
  CheckWhetherPreparePhiData(Inputs(), Outputs(), scope);
#if defined(PADDLE_WITH_XPU)
  if (std::getenv("XPU_NEED_PREPARE_PHI_DATA") != nullptr) {
    need_prepare_phi_data_ = atoi(std::getenv("XPU_NEED_PREPARE_PHI_DATA"));
  }
#endif
  if (!enable_cache_runtime_context_) {
    RuntimeContext ctx(Inputs(), Outputs(), scope);
    RunImpl(scope, place, &ctx);
  } else if (run_phi_kernel_ && impl_ != nullptr && !need_prepare_data_ &&
             !need_prepare_phi_data_) {
    if (!all_kernels_must_compute_runtime_shape_ && impl_->NeedInferShape()) {
      this->Info().infer_shape_(impl_->getRuntimeInferShapeContext());
    }
    (*phi_kernel_)(impl_->getKernelContext());
  } else {
    if (runtime_ctx_.get() == nullptr || pre_scope_ != cur_scope) {
      std::lock_guard<std::mutex> lock(cache_update_mutex_);
      if (runtime_ctx_.get() == nullptr || pre_scope_ != cur_scope) {
        runtime_ctx_ =
            std::make_unique<RuntimeContext>(Inputs(), Outputs(), scope);
        pre_scope_ = cur_scope;
      }
    }
    RunImpl(scope, place, runtime_ctx_.get());
  }
}

void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const phi::Place& place,
                                 RuntimeContext* runtime_ctx) const {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  bool fallback_to_cpu = false;
  phi::KernelKey phi_cpu_kernel_key;
  auto* dev_ctx = pool.Get(place);
  // using cache
  if (kernel_type_.get()) {
    dev_ctx = pool.Get(kernel_type_->place_);
  }
  auto exe_ctx = ExecutionContext(*this, scope, *dev_ctx, *runtime_ctx);

// TODO(Liu-xiandong): Now we are using too much if-else and hard code in XPU
// device, it's ugly, and we will refactor in the future.
#if defined(PADDLE_WITH_XPU_KP)
  bool use_phi_xpu_kp = false;
#endif

  // TODO(chenweihang): Now we are still reusing a lot of the original fluid
  // implementation, this is a gradual replacement process
  // TODO(chenweihang): in the first phase of project, we only support CPU, CUDA
  // and RCOM backend, the XPU, NPU and OneDNN will be supported in the second
  // phase
  phi::KernelKey phi_kernel_key;
  std::string phi_kernel_name;
  if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(type_)) {
    if (kernel_signature_ == nullptr || phi_kernel_ == nullptr) {
      if (phi::KernelFactory::Instance().HasStructuredKernel(
              type_)) {  // NOLINT
        kernel_signature_ =
            std::make_unique<phi::KernelSignature>(type_.c_str());
      } else {
        kernel_signature_ = std::make_unique<phi::KernelSignature>(
            GetExpectedPhiKernelArgs(exe_ctx));
      }

      VLOG(6) << *kernel_signature_.get();
      phi_kernel_name = kernel_signature_->name;
      kernel_type_ =
          std::make_unique<OpKernelType>(InnerGetExpectedKernelType(exe_ctx));
      dev_ctx = pool.Get(kernel_type_->place_);
// NOTE(Liu-xiandong): The register kernel used KP have library_type[KP],
// But the default library_type is Plain, so we need to modify the
// library_type here, otherwise it can't work.
#ifdef PADDLE_WITH_XPU_KP
      if (phi::is_xpu_place(kernel_type_->place_)) {
        bool use_xpu_kp_kernel_rt =
            FLAGS_run_kp_kernel &&
            paddle::platform::is_xpu_kp_support_op(
                type_, phi::TransToPhiDataType(kernel_type_->data_type_));
        bool use_xpu_kp_kernel_debug =
            paddle::platform::is_in_xpu_kpwhite_list(type_);
        if (use_xpu_kp_kernel_rt) {
          VLOG(3) << "phi xpu_kp using rt mode in static graph";
        }
        if (use_xpu_kp_kernel_debug) {
          VLOG(3) << "phi xpu_kp using debug mode in static graph";
        }
        bool is_xpu_kp_support =
            (use_xpu_kp_kernel_rt || use_xpu_kp_kernel_debug);
        if (is_xpu_kp_support) {
          auto expected_kernel_key_library_type = kernel_type_->library_type_;
          kernel_type_->library_type_ = LibraryType::kKP;
          VLOG(3) << "modifying XPU KP kernel in static graph: "
                  << phi_kernel_name
                  << ", using_kernel_key:" << *kernel_type_.get();
          auto try_phi_kernel_key =
              TransOpKernelTypeToPhiKernelKey(*kernel_type_.get());
          if (!phi::KernelFactory::Instance().HasKernel(phi_kernel_name,
                                                        try_phi_kernel_key)) {
            kernel_type_->library_type_ = expected_kernel_key_library_type;
            VLOG(3) << "modify XPU KP kernel in static graph: "
                    << phi_kernel_name << " is failed " << *kernel_type_.get();
          } else {
            use_phi_xpu_kp = true;
            VLOG(3) << "modify XPU KP kernel in static graph: "
                    << phi_kernel_name << " is succeed " << *kernel_type_.get();
          }
        }
      }
#endif
      phi_kernel_key = TransOpKernelTypeToPhiKernelKey(*kernel_type_.get());
      phi_kernel_ = std::make_unique<phi::Kernel>(
          phi::KernelFactory::Instance().SelectKernel(phi_kernel_name,
                                                      phi_kernel_key));

      if (phi_kernel_->IsValid()) {
        VLOG(6) << "Static graph mode ChoosePhiKernel - kernel name: "
                << phi_kernel_name << " | kernel key: " << phi_kernel_key
                << " | kernel: " << *phi_kernel_;
      } else {
        VLOG(6) << "Static graph mode ChoosePhiKernel - kernel `"
                << phi_kernel_name << "` not found.";
      }
    } else {
      phi_kernel_name = kernel_signature_->name;
// NOTE(jiahongyu): The registered OneDNN kernel have library_type =
// LibraryType::kMKLDNN and data_layout_ = DataLayout::ONEDNN. But the default
// values are kPlain, so we need to modify the library_type and data_layout_
// here. There are three statements in if condition:
// 1. Whether onednn kernel fallbacks to plain kernel;
// 2. Whether this op has specific implementation;
// 3. Whether onednn kernel can be used.
#ifdef PADDLE_WITH_DNNL
      if (!this->DnnFallback() &&
          !paddle::platform::in_onednn_white_list(type_) &&
          this->CanMKLDNNBeUsed(exe_ctx, kernel_type_->data_type_)) {
        kernel_type_->library_type_ = framework::LibraryType::kMKLDNN;
        kernel_type_->data_layout_ = framework::DataLayout::ONEDNN;
      } else if (phi::is_cpu_place(kernel_type_->place_) &&
                 kernel_type_->data_type_ ==
                     proto::VarType::Type::VarType_Type_BF16 &&
                 !this->SupportsCPUBF16() &&
                 this->SupportsMKLDNN(phi::DataType::BFLOAT16)) {
        kernel_type_->library_type_ = framework::LibraryType::kMKLDNN;
        kernel_type_->data_layout_ = framework::DataLayout::ONEDNN;
      }
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      if (this->CanCUDNNBeUsed(exe_ctx, kernel_type_->data_type_)) {
        kernel_type_->library_type_ = framework::LibraryType::kCUDNN;
      }
#endif

// NOTE(Liu-xiandong):In my ctest, this branch do not be executed,
// I can't understand it, it's really confusing.
// But we still need to keep this to avoid errors.
#ifdef PADDLE_WITH_XPU_KP
      if (phi::is_xpu_place(kernel_type_->place_)) {
        bool use_xpu_kp_kernel_rt =
            FLAGS_run_kp_kernel &&
            paddle::platform::is_xpu_kp_support_op(
                type_, phi::TransToPhiDataType(kernel_type_->data_type_));
        bool use_xpu_kp_kernel_debug =
            paddle::platform::is_in_xpu_kpwhite_list(type_);
        if (use_xpu_kp_kernel_rt) {
          VLOG(3) << "phi xpu_kp using rt mode in static graph";
        }
        if (use_xpu_kp_kernel_debug) {
          VLOG(3) << "phi xpu_kp using debug mode in static graph";
        }
        bool is_xpu_kp_support =
            (use_xpu_kp_kernel_rt || use_xpu_kp_kernel_debug);
        if (is_xpu_kp_support) {
          auto expected_kernel_key_library_type = kernel_type_->library_type_;
          kernel_type_->library_type_ = LibraryType::kKP;
          VLOG(3) << "modifying XPU KP kernel in static graph: "
                  << phi_kernel_name
                  << ", using_kernel_key:" << *kernel_type_.get();
          auto try_phi_kernel_key =
              TransOpKernelTypeToPhiKernelKey(*kernel_type_.get());
          if (!phi::KernelFactory::Instance().HasKernel(phi_kernel_name,
                                                        try_phi_kernel_key)) {
            kernel_type_->library_type_ = expected_kernel_key_library_type;
            VLOG(3) << "modify XPU KP kernel in static graph: "
                    << phi_kernel_name << " is failed " << *kernel_type_.get();
          } else {
            use_phi_xpu_kp = true;
            VLOG(3) << "modify XPU KP kernel in static graph: "
                    << phi_kernel_name << " is succeed " << *kernel_type_.get();
          }
        }
      }
#endif
      phi_kernel_key = TransOpKernelTypeToPhiKernelKey(*kernel_type_.get());
    }

// NOTE(Liu-xiandong): Determine whether the selected kernel is valid
// If not, use the kernel registered in fluid. And if the fluid do not
// contains the related heterogeneous kernel, use phi CPU kernel.
#if defined(PADDLE_WITH_XPU)
    bool is_xpu_unsupported =
        phi::is_xpu_place(kernel_type_->place_) &&
        !paddle::platform::is_xpu_support_op(
            type_, phi::TransToPhiDataType(kernel_type_->data_type_));
#endif
#ifdef PADDLE_WITH_XPU_KP
    bool use_xpu_kp_kernel_rt =
        phi::is_xpu_place(kernel_type_->place_) && FLAGS_run_kp_kernel &&
        paddle::platform::is_xpu_kp_support_op(
            type_, phi::TransToPhiDataType(kernel_type_->data_type_));
    bool use_xpu_kp_kernel_debug =
        phi::is_xpu_place(kernel_type_->place_) &&
        paddle::platform::is_in_xpu_kpwhite_list(type_);
    bool is_xpu_kp_support = (use_xpu_kp_kernel_rt || use_xpu_kp_kernel_debug);
#endif

    bool in_custom_back_list = false;
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
    in_custom_back_list =
        phi::backends::custom_device::is_in_custom_black_list(phi_kernel_name);
#endif
    if (phi_kernel_->IsValid() && !in_custom_back_list
#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
        && !is_xpu_unsupported
#endif
#if defined(PADDLE_WITH_XPU_KP)
        && (!is_xpu_unsupported || use_phi_xpu_kp)
#endif
    ) {
      run_phi_kernel_ = true;
    } else {
      auto& all_op_kernels = AllOpKernels();
      auto kernels_iter = all_op_kernels.find(type_);

// NOTE(Liu-xiandong): If we can't find heterogeneous kernel in phi,
// we need to select the heterogeneous kernel in fluid, but the kernel
// registered in KP use library_type[KP], we need to modify it.
#ifdef PADDLE_WITH_XPU_KP
      if (is_xpu_kp_support) {
        kernel_type_->library_type_ = LibraryType::kKP;
      }
#endif
      if (kernels_iter == all_op_kernels.end() ||
          kernels_iter->second.find(*kernel_type_.get()) ==
              kernels_iter->second.end()
#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
          || is_xpu_unsupported
#endif
#if defined(PADDLE_WITH_XPU_KP)
          || (is_xpu_unsupported && !is_xpu_kp_support)
#endif
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
          || in_custom_back_list
#endif
      ) {
        fallback_to_cpu = true;
        if (in_custom_back_list) {
          VLOG(3) << "fluid in black list: " << phi_kernel_name;
        }
        phi_cpu_kernel_key = FallBackToCpu(phi_kernel_key, *this);
        phi_kernel_ = std::make_unique<phi::Kernel>(
            phi::KernelFactory::Instance().SelectKernel(phi_kernel_name,
                                                        phi_cpu_kernel_key));

        dev_ctx = pool.Get(phi::CPUPlace());
        if (phi_kernel_->IsValid()) {
          VLOG(6) << "Static graph mode PrepareImpl - kernel name: "
                  << phi_kernel_name << " | kernel key: " << phi_cpu_kernel_key
                  << " | kernel: " << *phi_kernel_;
          run_phi_kernel_ = true;
        }
      }
    }
  }
  if (!run_phi_kernel_) {
    if (kernel_type_.get() == nullptr || kernel_func_.get() == nullptr) {
      ChooseKernel(exe_ctx);
      dev_ctx = pool.Get(kernel_type_->place_);
    }
  }

  // do data transformScope &transfer_scope;
  std::vector<std::string> transferred_inplace_vars;
  Scope* transfer_scope = nullptr;
  {
    phi::RecordEvent record_event("prepare_data",
                                  phi::TracerEventType::OperatorInner,
                                  1,
                                  phi::EventRole::kInnerOp);
    if (need_prepare_data_) {
      if (fallback_to_cpu) {  // NOLINT
        transfer_scope = PrepareData(scope,
                                     phi_cpu_kernel_key,
                                     &transferred_inplace_vars,
                                     runtime_ctx,
                                     dev_ctx->GetPlace());
      } else {
        transfer_scope = PrepareData(
            scope,
            framework::TransOpKernelTypeToPhiKernelKey(*kernel_type_),
            &transferred_inplace_vars,
            runtime_ctx,
            dev_ctx->GetPlace());
      }
    }
  }
  // exec scope is the scope that kernel actually executed on.
  const Scope& exec_scope =
      (transfer_scope == nullptr ? scope : *transfer_scope);

  if (!all_kernels_must_compute_runtime_shape_) {
    phi::RecordEvent record_event("infer_shape",
                                  phi::TracerEventType::OperatorInner,
                                  1,
                                  phi::EventRole::kInnerOp);
    RuntimeInferShapeContext infer_shape_ctx(*this, *runtime_ctx);
    this->Info().infer_shape_(&infer_shape_ctx);
    record_event.End();
    platform::RecordOpInfoSupplement(
        Type(), Attrs(), infer_shape_ctx, *runtime_ctx, Id());
  }

  if (FLAGS_enable_unused_var_check) {
    GetThreadLocalUsedVarNameSet()->clear();
  }

  // TODO(panyx0718): ExecutionContext should only depend on RuntimeContext
  // not Scope. Imperative mode only pass inputs and get outputs.
  {
    phi::RecordEvent record_event("compute",
                                  phi::TracerEventType::OperatorInner,
                                  1,
                                  phi::EventRole::kInnerOp);
    if (run_phi_kernel_ && phi_kernel_->GetKernelRegisteredType() ==
                               phi::KernelRegisteredType::FUNCTION) {
      phi::KernelContext phi_kernel_context;
      if (enable_cache_runtime_context_ && !need_prepare_phi_data_ &&
          !need_prepare_data_) {
        // TODO(inference): Now we only support dense_tensor cache, we may be
        // support ScalarTensor, SparseTensor in future.
        bool all_dense_tensor_input_{true};
        for (auto& iter : Inputs()) {
          for (auto& name : iter.second) {
            all_dense_tensor_input_ &=
                scope.FindVar(name)->IsType<phi::DenseTensor>();
          }
        }

        std::vector<phi::DenseTensor*> tensors;
        if (all_dense_tensor_input_) {
          for (auto& iter : Inputs()) {
            for (auto& name : iter.second) {
              auto* t = scope.FindVar(name)->GetMutable<phi::DenseTensor>();
              tensors.push_back(t);
            }
          }
        }

        impl_ = std::make_unique<CacheImpl>(
            new phi::KernelContext(),
            new RuntimeInferShapeContext(*this, *runtime_ctx),
            tensors,
            HasAttr(CacheImpl::kNotAllowInferShapeCache));
        BuildPhiKernelContext(*runtime_ctx, dev_ctx, impl_->getKernelContext());
        (*phi_kernel_)(impl_->getKernelContext());
      } else {
        phi::KernelContext phi_kernel_context;
        // Do data transform before building KernelContext
        // TODO(zhiqiu): support TransferInplaceVarsBack
        BuildPhiKernelContext(*runtime_ctx, dev_ctx, &phi_kernel_context);
        (*phi_kernel_)(&phi_kernel_context);
      }
    } else if (run_phi_kernel_ && phi_kernel_->GetKernelRegisteredType() ==
                                      phi::KernelRegisteredType::STRUCTURE) {
      ExecutionContext execution_context(
          *this, exec_scope, *dev_ctx, *runtime_ctx);
      (*phi_kernel_)(&execution_context);
    } else {
      (*kernel_func_)(
          ExecutionContext(*this, exec_scope, *dev_ctx, *runtime_ctx));
    }
    if (fallback_to_cpu) {
      phi_kernel_.reset();
    }
  }

  if (!transferred_inplace_vars.empty()) {
    // there is inplace variable has been transferred.
    TransferInplaceVarsBack(scope, transferred_inplace_vars, *transfer_scope);
  }

  // See [ Why need handle complex gradient to real gradient? ]
  // Only handle the case where the current kernel data type is complex
  if (framework::IsComplexType(kernel_type_->data_type_)) {
    HandleComplexGradToRealGrad(scope, runtime_ctx);
  }

  if (FLAGS_enable_unused_var_check) {
    // skip op that uses onednn because it has different memory reuse strategy.
    // use attr here because some GradMakers (like ActivationGradOpMaker) add
    // input when use_mkldnn=true;
    if (!(HasAttr("use_mkldnn") && Attr<bool>("use_mkldnn"))) {
      CheckUnusedVar(*this, scope);
    }
  }

  /*For profiling/benchmark only*/
  if (FLAGS_benchmark) {
    dev_ctx->Wait();
#if defined(PADDLE_WITH_CUDA) || defined(PADLDE_WITH_ROCM)
    PADDLE_ENFORCE_GPU_SUCCESS(platform::GpuGetLastError());
#endif
    VLOG(4) << "Operator(" << Type() << "): context wait and get last error";
  }

  if (FLAGS_check_nan_inf) {
    try {
      framework::details::CheckOpHasNanOrInf(*this, exec_scope, place);
    } catch (...) {
      const std::vector<std::string>* callstack = nullptr;
      auto attrs = Attrs();
      auto iter =
          attrs.find(OpProtoAndCheckerMaker::OpCreationCallstackAttrName());
      if (iter != attrs.end()) {
        callstack = &PADDLE_GET_CONST(std::vector<std::string>, iter->second);
        if (callstack->empty()) callstack = nullptr;
      }
      std::ostringstream sout;
      if (callstack) {
        if (FLAGS_call_stack_level > 1) {
          sout << "\n\n  Compile Traceback (most recent call last):";
        } else {
          sout << "In user code:\n";
        }
        for (auto& line : *callstack) {
          sout << "\n  " << line;
        }
      }
      std::cout << sout.str() << std::endl;
      std::rethrow_exception(std::current_exception());
    }
  }

  // To solve issue #15032, have a discussion with @Luotao for cpu inference,
  // do not cache transfer scope, hence in this case delete transfer scope
  // after run to avoid memory leak
  if (transfer_scope && !run_by_executor_ && !enable_cache_transfer_scope_) {
    scope.DeleteScope(transfer_scope);
  }
}

OpKernelType OperatorWithKernel::InnerGetExpectedKernelType(
    const ExecutionContext& ctx) const {
  phi::KernelKey phi_kernel_key = this->GetExpectedKernelType(ctx);
  auto expected_kernel_key =
      framework::TransPhiKernelKeyToOpKernelType(phi_kernel_key);

// NOTE(jiahongyu): PADDLE_WITH_DNNL codes are moved outside function
// GetExpectedKernelType, so that if OneDNN can be used, the library_type_ and
// data_layout_ of expected_kernel_key need to be adjusted. There are three
// statements in if condition:
// 1. Whether onednn kernel fallbacks to plain kernel;
// 2. Whether this op has specific implementation;
// 3. Whether onednn kernel can be used.
#ifdef PADDLE_WITH_DNNL
  if (!this->DnnFallback() && !paddle::platform::in_onednn_white_list(type_) &&
      this->CanMKLDNNBeUsed(ctx, expected_kernel_key.data_type_)) {
    expected_kernel_key.library_type_ = framework::LibraryType::kMKLDNN;
    expected_kernel_key.data_layout_ = framework::DataLayout::ONEDNN;
  } else if (phi::is_cpu_place(expected_kernel_key.place_) &&
             expected_kernel_key.data_type_ ==
                 proto::VarType::Type::VarType_Type_BF16 &&
             !this->SupportsCPUBF16() &&
             this->SupportsMKLDNN(phi::DataType::BFLOAT16)) {
    expected_kernel_key.library_type_ = framework::LibraryType::kMKLDNN;
    expected_kernel_key.data_layout_ = framework::DataLayout::ONEDNN;
  }
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (this->CanCUDNNBeUsed(ctx, expected_kernel_key.data_type_)) {
    expected_kernel_key.library_type_ = framework::LibraryType::kCUDNN;
  }
#endif

  if (HasAttr("op_device")) {
    if (Attr<std::string>("op_device") == "cpu") {
      expected_kernel_key.place_ = phi::CPUPlace();
    } else if (Attr<std::string>("op_device").find("gpu") !=
               std::string::npos) {
      auto device = Attr<std::string>("op_device");
      size_t pos = device.find(':');
      if (pos != std::string::npos) {
        device = device.substr(0, pos);
        LOG_FIRST_N(WARNING, 1)
            << "Device index is only supported under pipeline parallelism, "
            << "so it will be ignored.";
      }
      // when the Op that does not have GPUKernel is assigned to GPU, the
      // CPUKernel will be executed and a warning will be given at the same
      // time.
      expected_kernel_key.place_ = phi::CPUPlace();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      if (SupportGPU()) {
        auto& dev_ctx = ctx.device_context();
        expected_kernel_key.place_ = dev_ctx.GetPlace();
      }
#endif
      if (phi::is_cpu_place(expected_kernel_key.place_)) {
        LOG_FIRST_N(WARNING, 1)
            << "Op(" << type_
            << ") has no CUDA implementation. It will be assigned to CPUPlace.";
      }
    } else if (Attr<std::string>("op_device").find("npu") !=
               std::string::npos) {
      auto device = Attr<std::string>("op_device");
      size_t pos = device.find(':');
      if (pos != std::string::npos) {
        device = device.substr(0, pos);
        LOG_FIRST_N(WARNING, 1)
            << "Device index is only supported under pipeline parallelism, "
            << "so it will be ignored.";
      }
      // when the Op that does not have NPUKernel is assigned to NPU, the
      // CPUKernel will be executed and a warning will be given at the same
      // time.
      expected_kernel_key.place_ = phi::CPUPlace();
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      if (SupportCustomDevice()) {
        auto& dev_ctx = ctx.device_context();
        expected_kernel_key.place_ = dev_ctx.GetPlace();
      }
#endif
      if (phi::is_cpu_place(expected_kernel_key.place_)) {
        LOG_FIRST_N(WARNING, 1)
            << "Op(" << type_
            << ") has no NPU implementation. It will be assigned to CPUPlace.";
      }
    } else if (Attr<std::string>("op_device").find("xpu") !=
               std::string::npos) {
      auto device = Attr<std::string>("op_device");
      size_t pos = device.find(':');
      if (pos != std::string::npos) {
        device = device.substr(0, pos);
        LOG_FIRST_N(WARNING, 1)
            << "Device index is only supported under pipeline parallelism, "
            << "so it will be ignored.";
      }
      // when the Op that does not have XPUKernel is assigned to XPU, the
      // CPUKernel will be executed and a warning will be given at the same
      // time.
      expected_kernel_key.place_ = phi::CPUPlace();
#ifdef PADDLE_WITH_XPU
      if (SupportXPU()) {
        auto& dev_ctx = ctx.device_context();
        expected_kernel_key.place_ = dev_ctx.GetPlace();
      }
#endif
      if (phi::is_cpu_place(expected_kernel_key.place_)) {
        LOG_FIRST_N(WARNING, 1)
            << "Op(" << type_
            << ") has no XPU implementation. It will be assigned to CPUPlace.";
      }
    }
  }

  if (phi::places_are_same_class(expected_kernel_key.place_, ctx.GetPlace())) {
    expected_kernel_key.place_ = ctx.GetPlace();
  }

  VLOG(3) << "op type:" << type_
          << ", expected_kernel_key:" << expected_kernel_key;
  return expected_kernel_key;
}

phi::KernelKey OperatorWithKernel::ChoosePhiKernel(
    const ExecutionContext& ctx) const {
  std::string phi_kernel_name;
  if (phi::KernelFactory::Instance().HasStructuredKernel(type_)) {  // NOLINT
    kernel_signature_ = std::make_unique<phi::KernelSignature>(type_.c_str());
  } else {
    kernel_signature_ =
        std::make_unique<phi::KernelSignature>(GetExpectedPhiKernelArgs(ctx));
  }
  VLOG(6) << *kernel_signature_.get();
  phi_kernel_name = kernel_signature_->name;
  kernel_type_ =
      std::make_unique<OpKernelType>(InnerGetExpectedKernelType(ctx));

  auto phi_kernel_key = TransOpKernelTypeToPhiKernelKey(*kernel_type_.get());
  phi_kernel_ =
      std::make_unique<phi::Kernel>(phi::KernelFactory::Instance().SelectKernel(
          phi_kernel_name, phi_kernel_key));

  if (phi_kernel_->IsValid()) {
    VLOG(6) << "Static graph mode ChoosePhiKernel - kernel name: "
            << phi_kernel_name << " | kernel key: " << phi_kernel_key
            << " | kernel: " << *phi_kernel_;
  } else {
    VLOG(6) << "Static graph mode ChoosePhiKernel - kernel `" << phi_kernel_name
            << "` not found.";
  }
  return phi_kernel_key;
}

void OperatorWithKernel::ChooseKernel(const ExecutionContext& ctx) const {
  // check if op[type] has kernel registered.
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  PADDLE_ENFORCE_NE(
      kernels_iter,
      all_op_kernels.end(),
      common::errors::Unimplemented(
          "There are no kernels which are registered in the %s operator.",
          type_));

  OpKernelMap& kernels = kernels_iter->second;

  auto expected_kernel_key = InnerGetExpectedKernelType(ctx);

  auto kernel_iter = kernels.find(expected_kernel_key);

#ifdef PADDLE_WITH_DNNL
  // workaround for missing MKLDNN kernel when FLAGS_use_mkldnn env var is set
  if (kernel_iter == kernels.end() &&
      expected_kernel_key.library_type_ == LibraryType::kMKLDNN) {
    VLOG(3) << "missing MKLDNN kernel: fallbacking to PLAIN one";
    expected_kernel_key.library_type_ = LibraryType::kPlain;
    expected_kernel_key.data_layout_ = DataLayout::kAnyLayout;
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif

#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
  if (phi::is_xpu_place(expected_kernel_key.place_) &&
      (kernel_iter == kernels.end() ||
       !paddle::platform::is_xpu_support_op(
           type_, phi::TransToPhiDataType(expected_kernel_key.data_type_)))) {
    VLOG(3) << "fluid missing XPU kernel: " << type_
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    expected_kernel_key.place_ = phi::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif

#ifdef PADDLE_WITH_XPU_KP
  if (phi::is_xpu_place(expected_kernel_key.place_)) {
    bool use_xpu_kp_kernel_rt =
        FLAGS_run_kp_kernel &&
        paddle::platform::is_xpu_kp_support_op(
            type_, phi::TransToPhiDataType(expected_kernel_key.data_type_));
    bool use_xpu_kp_kernel_debug =
        paddle::platform::is_in_xpu_kpwhite_list(type_);
    if (use_xpu_kp_kernel_rt) {
      VLOG(3) << "fluid xpu_kp using rt mode ";
    }
    if (use_xpu_kp_kernel_debug) {
      VLOG(3) << "fluid xpu_kp using debug mode ";
    }
    bool is_xpu_kp_support = (use_xpu_kp_kernel_rt || use_xpu_kp_kernel_debug);
    if (is_xpu_kp_support) {
      auto cache_expected_kernel_key_library_type =
          expected_kernel_key.library_type_;
      expected_kernel_key.library_type_ = LibraryType::kKP;
      kernel_iter = kernels.find(expected_kernel_key);
      // if can't find corresponding kernel when is_xpu_kp_support is on
      // if the fluid do not register related kernel, it can't work and have
      // error as before
      if (kernel_iter == kernels.end()) {
        expected_kernel_key.library_type_ =
            cache_expected_kernel_key_library_type;
        expected_kernel_key.place_ = phi::CPUPlace();
        kernel_iter = kernels.find(expected_kernel_key);
      } else {
        VLOG(3) << "fluid using XPU KP kernel: " << type_
                << ", using_kernel_key:" << expected_kernel_key;
      }
    }
    bool is_xpu_unsupported = (!paddle::platform::is_xpu_support_op(
        type_, phi::TransToPhiDataType(expected_kernel_key.data_type_)));
    if (!is_xpu_kp_support &&
        (kernel_iter == kernels.end() || is_xpu_unsupported)) {
      VLOG(3) << "fluid missing XPU kernel: " << type_
              << ", expected_kernel_key:" << expected_kernel_key
              << ", fallbacking to CPU one!";
      expected_kernel_key.place_ = phi::CPUPlace();
      kernel_iter = kernels.find(expected_kernel_key);
    }
  }
#endif

#ifdef PADDLE_WITH_IPU
  if (kernel_iter == kernels.end() &&
      phi::is_ipu_place(expected_kernel_key.place_)) {
    VLOG(3) << "missing IPU kernel: " << type_
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    expected_kernel_key.place_ = phi::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (kernel_iter == kernels.end() &&
      phi::is_custom_place(expected_kernel_key.place_)) {
    VLOG(3) << "missing " << expected_kernel_key.place_.GetDeviceType()
            << " kernel: " << type_
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    expected_kernel_key.place_ = phi::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
  PADDLE_ENFORCE_NE(
      kernel_iter,
      kernels.end(),
      common::errors::NotFound("Operator (%s) does not have kernel for %s.",
                               type_,
                               KernelTypeToString(expected_kernel_key)));

  std::lock_guard<std::mutex> lock(cache_update_mutex_);
  if (kernel_type_.get() == nullptr || kernel_func_.get() == nullptr) {
    kernel_type_ = std::make_unique<OpKernelType>(expected_kernel_key);
    kernel_func_ = std::make_unique<OpKernelFunc>(kernel_iter->second);
  }
}

void OperatorWithKernel::TransferInplaceVarsBack(
    const Scope& scope,
    const std::vector<std::string>& inplace_vars,
    const Scope& transfer_scope) const {
  for (auto& var_name : inplace_vars) {
    VLOG(3) << "share inplace var " + var_name + " back to it's original scope";
    auto* origin_var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(origin_var,
                            common::errors::InvalidArgument(
                                "The variable[%s] is nullptr.", var_name));
    auto* original_tensor =
        GetMutableDenseTensorOrSelectedRowsValueFromVar(origin_var);
    auto* var = transfer_scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(var,
                            common::errors::InvalidArgument(
                                "The variable[%s] is nullptr.", var_name));
    auto* transformed_tensor = GetDenseTensorOrSelectedRowsValueFromVar(*var);
    original_tensor->ShareDataWith(*transformed_tensor);
  }
}

void OperatorWithKernel::HandleComplexGradToRealGrad(
    const Scope& scope, RuntimeContext* ctx) const {
  for (auto& var_name_item : Outputs()) {
    std::vector<Variable*>& output_vars = ctx->outputs[var_name_item.first];
    for (size_t i = 0; i < var_name_item.second.size(); ++i) {
      // 1. find grad_var & check whether is complex tensor
      auto var_name = var_name_item.second[i];
      auto orig_var_name = GradOriginalVarName(var_name);
      // only focus on gradient var
      if (var_name == orig_var_name) {
        continue;
      }
      auto* grad_var = output_vars[i];
      // skip nullptr var
      if (grad_var == nullptr) {
        continue;
      }
      // don't process phi::TensorArray temporarily,
      // add support if necessary for complex number calculations in the future
      if (!VarIsTensor(*grad_var)) {
        continue;
      }
      auto* grad_tensor =
          GetMutableDenseTensorOrSelectedRowsValueFromVar(grad_var);
      // skip nullptr tensor
      if (grad_tensor == nullptr || !grad_tensor->IsInitialized()) {
        continue;
      }
      // only focus on complex dtype now
      auto src_type = framework::TransToProtoVarType(grad_tensor->dtype());
      if (!IsComplexType(src_type)) {
        continue;
      }

      // 2. find forward var & check whether need to cast
      auto* var = scope.FindVar(orig_var_name);
      // if forward var not exists, do nothing
      if (var == nullptr) {
        continue;
      }
      if (!VarIsTensor(*var)) {
        continue;
      }
      const auto* tensor = GetDenseTensorOrSelectedRowsValueFromVar(*var);
      PADDLE_ENFORCE_NOT_NULL(
          tensor,
          common::errors::Unavailable(
              "Forward tensor is nullptr when handle complex data to real."));
      // only need record type, the allocation may have been released
      auto dst_type = framework::TransToProtoVarType(tensor->dtype());
      // only focus on real dtype and need casting
      if (IsComplexType(dst_type)) {
        continue;
      }

      // 3. cast complex grad to real grad
      VLOG(6) << "Transform " << framework::DataTypeToString(src_type)
              << " var `" << var_name << "` to "
              << framework::DataTypeToString(dst_type)
              << " real var in static graph.";
      phi::DenseTensor out;
      TransComplexToReal(dst_type, src_type, *grad_tensor, &out);
      SetTensorToVariable(*grad_var, out, grad_var);
    }
  }
}

Scope* OperatorWithKernel::PrepareData(
    const Scope& scope,
    const phi::KernelKey& expected_kernel_key,
    std::vector<std::string>* transferred_inplace_vars,
    RuntimeContext* ctx,
    const phi::Place& place) const {
  Scope* new_scope = nullptr;

  const std::unordered_set<std::string>* no_buffer_ins = nullptr;
  if (info_) {
    auto& no_buffer_inferer = info_->NoNeedBufferVarsInferer();
    // Some op may not register NoNeedBufferVarsInferer
    if (no_buffer_inferer) {
      no_buffer_ins = &(no_buffer_inferer(Inputs(), Outputs(), Attrs()));
      if (no_buffer_ins->empty()) no_buffer_ins = nullptr;
    }
  }

  auto has_infer_varkernel_fn =
      (run_phi_kernel_ && phi_kernel_->get_kerneltype_forvar_fn_ != nullptr);
  phi::AttributeMap infer_attrs{};
  auto fluid_attrs = Attrs();
  phi::GetKernelTypeForVarContext infer_varkernel_context =
      BuildGetKernelTypeForVarContext(expected_kernel_key,
                                      fluid_attrs,
                                      &infer_attrs,
                                      has_infer_varkernel_fn);

  const auto& name_map = Inputs();
  auto prepare_input_data = [&](const std::string& in_name,
                                std::vector<Variable*>* in_vars,
                                const phi::TensorArgDef* in_def,
                                bool should_skip_input) -> void {
    auto& name_vec = name_map.at(in_name);
    for (size_t i = 0; i < in_vars->size(); ++i) {
      const auto& var_name = name_vec[i];
      auto* var = in_vars->at(i);

      // Only tensor can be transfer to another device.
      if (var == nullptr || !VarIsTensor(*var)) {
        continue;
      }

      auto* tensor_in = GetDenseTensorOrSelectedRowsValueFromVar(*var);

      // When no_buffer_ins then checking of phi::DenseTensor::holder_ is
      // not a thread safe. And for infershape scenario checks
      // to be omitted are not really needed
      if (should_skip_input == true) {
#ifdef PADDLE_WITH_DNNL
        // Var without buffer may be needed
        // for some situation like InferShape().
        // In this situation We cannot skip Var analysis, as
        // oneDNN shape of Var may differ from kNHWC Var
        // In such situation corresponding resized Var
        // has to be created and registered
        if ((tensor_in->layout() == DataLayout::ONEDNN) &&
            (var->IsType<phi::DenseTensor>() == true) &&
            (expected_kernel_key.layout() != DataLayout::ONEDNN) &&
            (phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
             DataLayout::kNHWC) &&
            (tensor_in->dims().size() >= 3)) {
          // Mixed execution : oneDNN and GPU is not supported!
          if (!new_scope) {
            new_scope = &scope.NewScope();
          }
          auto* trans_var = new_scope->Var(var_name);
          in_vars->at(i) = trans_var;
          auto out = trans_var->GetMutable<phi::DenseTensor>();
          out->Resize(tensor_in->dims());
          phi::funcs::MatchShapeToLayout(
              out, tensor_in->layout(), DataLayout::kNHWC);
          VLOG(7) << "Created reshaped dummy input based on oneDNN "
                     "phi::DenseTensor , "
                     "but kNHWC layout"
                  << in_name << " in Operator " << type_;
        } else {
          VLOG(7) << "Skip scanning input " << in_name << " in Operator "
                  << type_;
        }
#endif
        continue;
      }

      if (!tensor_in->IsInitialized()) {
        continue;
      }

      auto kernel_type_for_var =
          GetKernelTypeForVar(in_name, *tensor_in, expected_kernel_key);
      if (has_infer_varkernel_fn) {
        infer_varkernel_context.SetVarName(const_cast<std::string*>(&in_name));
        infer_varkernel_context.SetDenseTensor(
            const_cast<phi::DenseTensor*>(tensor_in));
        kernel_type_for_var =
            phi_kernel_->get_kerneltype_forvar_fn_(&infer_varkernel_context);
      }
      bool need_trans_dtype =
          NeedTransformDataType(expected_kernel_key, kernel_type_for_var);
      bool need_trans_layout = NeedTransformLayout(
          kernel_type_for_var.layout(), expected_kernel_key.layout());
      if (!need_trans_dtype && !need_trans_layout) {
        if (!run_phi_kernel_ &&
            backends_are_same_class(kernel_type_for_var.backend(),
                                    expected_kernel_key.backend())) {
          continue;
        }
      }

      std::unique_ptr<phi::KernelKey> new_expected_kernel_key = nullptr;
      if (run_phi_kernel_) {
        if (phi_kernel_->GetKernelRegisteredType() ==
            phi::KernelRegisteredType::STRUCTURE) {
          if (!backends_are_same_class(kernel_type_for_var.backend(),
                                       expected_kernel_key.backend())) {
            new_expected_kernel_key =
                std::make_unique<phi::KernelKey>(expected_kernel_key.backend(),
                                                 expected_kernel_key.layout(),
                                                 expected_kernel_key.dtype());
          }
        } else if (in_def != nullptr &&  // KernelRegisteredType is Function
                   in_def->backend != phi::Backend::ALL_BACKEND &&
                   kernel_type_for_var.backend() != phi::Backend::ALL_BACKEND) {
          auto tensor_backend = phi::TransToPhiBackend(tensor_in->place());
          if ((in_def->backend != tensor_backend &&
               !(in_def->backend == phi::Backend::GPUDNN &&
                 tensor_backend == phi::Backend::GPU) &&
               !(in_def->backend == phi::Backend::KPS &&
                 tensor_backend == phi::Backend::XPU) &&
               !(in_def->backend == phi::Backend::ONEDNN &&
                 tensor_backend == phi::Backend::CPU)) ||
              tensor_in->place().GetType() == AllocationType::GPUPINNED) {
            new_expected_kernel_key =
                std::make_unique<phi::KernelKey>(in_def->backend,
                                                 expected_kernel_key.layout(),
                                                 expected_kernel_key.dtype());
          }
        }
      }

      if (!need_trans_dtype && !need_trans_layout) {
        if (run_phi_kernel_ && new_expected_kernel_key == nullptr) {
          continue;
        }
      }

      VLOG(3) << "Transform Variable " << var_name << " from "
              << kernel_type_for_var << " to "
              << (new_expected_kernel_key ? *new_expected_kernel_key
                                          : expected_kernel_key);

      // In the inference scenario, the scopes will be reused across the
      // batches, so the `new_scope` here will result in GPU memory explosion
      // over the running of operators.
      // We use a thread_local cache to fix that issue, the key in the cache is
      // the combination of the `scope` argument, from_kernel_type,
      // target_kernel_type.
      // Have a discussion with @Superjomn or the inference developers if some
      // changes on this logic for this macro might not tested on the other
      // scenarios.
      // If this op is not called by an Executor or ParallelExecutor, it should
      // called by a NaiveExecutor, the NaiveExecutor will cache the scopes and
      // variables, that behavior a lot different.
      //
      // To solve issue #15032, have a discussion with @Luotao for cpu
      // inference, for all cpu kernels cases without GPU participation, here
      // not do transfer scope caching, and cpu inference performance is not
      // impacted by test.
      enable_cache_transfer_scope_ = false;
      if (!run_by_executor_) {
        if (new_expected_kernel_key) {
          if (kernel_type_for_var.backend() == phi::Backend::GPU ||
              kernel_type_for_var.backend() == phi::Backend::GPUDNN ||
              new_expected_kernel_key->backend() == phi::Backend::GPU ||
              new_expected_kernel_key->backend() == phi::Backend::GPUDNN ||
              kernel_type_for_var.backend() == phi::Backend::XPU ||
              new_expected_kernel_key->backend() == phi::Backend::XPU) {
            new_scope = TryCreateTransferScope(
                kernel_type_for_var, *new_expected_kernel_key, &scope);
            enable_cache_transfer_scope_ = true;
          }
        } else if (kernel_type_for_var.backend() == phi::Backend::GPU ||
                   kernel_type_for_var.backend() == phi::Backend::GPUDNN ||
                   expected_kernel_key.backend() == phi::Backend::GPU ||
                   expected_kernel_key.backend() == phi::Backend::GPUDNN ||
                   kernel_type_for_var.backend() == phi::Backend::XPU ||
                   expected_kernel_key.backend() == phi::Backend::XPU) {
          new_scope = TryCreateTransferScope(
              kernel_type_for_var, expected_kernel_key, &scope);
          enable_cache_transfer_scope_ = true;
        }
      }

      if (!new_scope) {
        new_scope = &scope.NewScope();
      }
      // For inference, if a gpu model has an op which could only run on CPU,
      // each result of different input will be the same with the first one.
      // The reason is that if a gpu tensor is the input of a cpu kernel,
      // we will create a new cpu tensor in new scope.
      // However, if enable_cache_runtime_context_, we get the cpu tensor each
      // time, not the gpu tensor. Thus, we set pre_scope_ = nullptr
      // to trigger `new RuntimeContext()` in RunImpl().
      if (enable_cache_runtime_context_) {
        pre_scope_ = nullptr;
      }

      // Create new var with the same name in transfer scopes
      auto* trans_var = new_scope->Var(var_name);
      in_vars->at(i) = trans_var;

      // Find if inplace exists between input and output
      // If inplace exists, set the new created var to inplaced output, and
      // record its name in transferred_inplace_vars.
      for (auto& pair : Outputs()) {
        for (size_t j = 0; j < pair.second.size(); ++j) {
          if (pair.second[j] == var_name) {
            VLOG(4) << "Found inplace between input(" << in_name
                    << ") and output(" << pair.first
                    << "), the variable name is " << var_name;
            ctx->outputs[pair.first][j] = trans_var;
            transferred_inplace_vars->emplace_back(var_name);
          }
        }
      }

      // Do transfer
      phi::DenseTensor out;
      TransformData(
          new_expected_kernel_key ? *new_expected_kernel_key
                                  : expected_kernel_key,
          kernel_type_for_var,
          *tensor_in,
          &out,
          new_expected_kernel_key
              ? phi::TransToPhiPlace(new_expected_kernel_key->backend())
              : place);
      SetTensorToVariable(*var, out, trans_var);
    }
  };

  if (run_phi_kernel_ && phi_kernel_->GetKernelRegisteredType() ==
                             phi::KernelRegisteredType::FUNCTION) {
    const auto& input_names = kernel_signature_->input_names;
    const auto& input_defs = phi_kernel_->args_def().input_defs();
    PADDLE_ENFORCE_EQ(input_names.size(),
                      input_defs.size(),
                      common::errors::InvalidArgument(
                          "The size of inputs_args names (%d) must be equal to "
                          "the size of kernel input_defs (%d).",
                          input_names.size(),
                          input_defs.size()));
    for (size_t i = 0; i < input_defs.size(); ++i) {
      std::string input_name = input_names[i];
      auto iter = ctx->inputs.find(input_name);
      if (iter == ctx->inputs.end()) {
        continue;
      }
      auto& ins_vector = iter->second;
      bool should_skip_input =
          no_buffer_ins && no_buffer_ins->count(input_name) > 0;

      phi::TensorArgDef in_def = input_defs.at(i);
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      // When the backend of input tensor arg_def is CUSTOM, we need to set it
      // to the actual backend by expected_kernel_key.
      if (in_def.backend == phi::Backend::CUSTOM) {
        in_def.SetBackend(expected_kernel_key.backend());
      }
#endif
      prepare_input_data(input_name, &ins_vector, &in_def, should_skip_input);
    }
#ifdef PADDLE_WITH_DNNL
    // For input that is Extra, only OneDNN will use Extra Inputs
    auto& extra_input_names =
        paddle::operators::ExtraInfoUtils::Instance().GetExtraInputNamesMap(
            Type());
    for (const auto& input_name : extra_input_names) {
      auto iter = ctx->inputs.find(input_name);
      if (iter == ctx->inputs.end()) {
        continue;
      }
      bool should_skip_input =
          no_buffer_ins && no_buffer_ins->count(input_name) > 0;
      std::vector<Variable*>& input_vars = iter->second;
      prepare_input_data(input_name, &input_vars, nullptr, should_skip_input);
    }
#endif
  } else {
    for (auto& var_name_item : Inputs()) {
      bool should_skip_input =
          no_buffer_ins && no_buffer_ins->count(var_name_item.first) > 0;

      std::vector<Variable*>& input_vars = ctx->inputs[var_name_item.first];
      prepare_input_data(
          var_name_item.first, &input_vars, nullptr, should_skip_input);
    }
  }

  // If pre_scope = &scope, it means that scope is cached and the op is not in
  // while block. If new_scope = nullptr, it means that for each input of this
  // Op, there is no need to do PrepareData. So PrepareData could be skipped at
  // the rest iterations to save the elapsed time.
  // We do not support skipping PrepareData in while block, because the Op's
  // input may be changed by subsequent Ops, which may cause an error.

  // For inference, ops that behind conditional branch aren't supported well,
  // so disable prepare optimization conservatively.
  bool force_prepare_data = HasAttr("inference_force_prepare_data") &&
                            Attr<bool>("inference_force_prepare_data");
  if (pre_scope_ == &scope && new_scope == nullptr && !force_prepare_data) {
    need_prepare_data_ = false;
  }

  return new_scope;
}

void OperatorWithKernel::ParseInputDataType(
    const Variable* var,
    const std::string& name,
    proto::VarType::Type* data_type) const {
  if (var != nullptr) {
    const phi::DenseTensor* t = nullptr;
    if (var->IsType<phi::DenseTensor>()) {
      t = &var->Get<phi::DenseTensor>();
    } else if (var->IsType<phi::SelectedRows>()) {
      t = &(var->Get<phi::SelectedRows>().value());
    } else if (var->IsType<phi::SparseCooTensor>()) {
      const phi::SparseCooTensor* sp_t = &(var->Get<phi::SparseCooTensor>());
      *data_type = paddle::framework::TransToProtoVarType(sp_t->dtype());
      return;
    } else if (var->IsType<phi::TensorArray>()) {
      auto t_arr = &var->Get<phi::TensorArray>();
      for (const auto& item : *t_arr) {
        if (item.IsInitialized()) {
          t = &(item);
        }
      }
    }
    if (t != nullptr) {
      *data_type = paddle::framework::TransToProtoVarType(t->dtype());
    }
  }
}

void OperatorWithKernel::ParseMultiInputDataType(
    const std::vector<Variable*>& vars,
    const std::string& name,
    proto::VarType::Type* data_type) const {
  proto::VarType::Type default_data_type =
      static_cast<proto::VarType::Type>(-1);
  for (auto* var : vars) {
    if (var != nullptr) {
      const phi::DenseTensor* t = nullptr;
      if (var->IsType<phi::DenseTensor>()) {
        t = &var->Get<phi::DenseTensor>();
      } else if (var->IsType<phi::SelectedRows>()) {
        t = &(var->Get<phi::SelectedRows>().value());
      } else if (var->IsType<phi::SparseCooTensor>()) {
        const phi::SparseCooTensor* sp_t = &(var->Get<phi::SparseCooTensor>());
        PADDLE_ENFORCE_EQ(
            sp_t->initialized(),
            true,
            common::errors::InvalidArgument("The %s Op's Input Variable `%s` "
                                            "contains uninitialized Tensor.",
                                            Type(),
                                            name));
        proto::VarType::Type tmp =
            paddle::framework::TransToProtoVarType(sp_t->dtype());
        PADDLE_ENFORCE(tmp == *data_type || *data_type == default_data_type,
                       common::errors::InvalidArgument(
                           "The DataType of %s Op's duplicable or different "
                           "slot Variable %s must be "
                           "consistent or register GetExpectedKernelType. The "
                           "current variable type is (%s), but the "
                           "previous variable type is (%s).",
                           Type(),
                           name,
                           DataTypeToString(tmp),
                           DataTypeToString(*data_type)));
        *data_type = tmp;
      } else if (var->IsType<phi::TensorArray>()) {
        auto t_arr = &var->Get<phi::TensorArray>();
        for (const auto& item : *t_arr) {
          if (item.IsInitialized()) {
            t = &(item);
          }
        }
      }
      if (t != nullptr) {
        PADDLE_ENFORCE_EQ(t->IsInitialized(),
                          true,
                          common::errors::InvalidArgument(
                              "The %s Op's Input Variable `%s` "
                              "contains uninitialized phi::DenseTensor.",
                              Type(),
                              name));
        proto::VarType::Type tmp =
            paddle::framework::TransToProtoVarType(t->dtype());
        PADDLE_ENFORCE(tmp == *data_type || *data_type == default_data_type,
                       common::errors::InvalidArgument(
                           "The DataType of %s Op's duplicable or different "
                           "slot Variable %s must be "
                           "consistent or register GetExpectedKernelType. The "
                           "current variable type is (%s), but the "
                           "previous variable type is (%s).",
                           Type(),
                           name,
                           DataTypeToString(tmp),
                           DataTypeToString(*data_type)));
        *data_type = tmp;
      }
    }
  }
}

proto::VarType::Type OperatorWithKernel::IndicateDataType(
    const ExecutionContext& ctx) const {
  proto::VarType::Type default_data_type =
      static_cast<proto::VarType::Type>(-1);
  proto::VarType::Type data_type = default_data_type;

  for (auto* name : ctx.InNameList()) {
    if (ctx.InputSize(*name) == 1UL) {
      ParseInputDataType(ctx.InputVar(*name), *name, &data_type);
    } else {
      ParseMultiInputDataType(ctx.MultiInputVar(*name), *name, &data_type);
    }
  }
  PADDLE_ENFORCE_NE(
      data_type,
      default_data_type,
      common::errors::NotFound(
          "DataType should be indicated by input Variable at %s.", Type()));
  return data_type;
}

proto::VarType::Type OperatorWithKernel::IndicateVarDataType(
    const ExecutionContext& ctx, const std::string& name) const {
  proto::VarType::Type default_data_type =
      static_cast<proto::VarType::Type>(-1);
  proto::VarType::Type data_type = default_data_type;
  if (ctx.InputSize(name) == 1UL) {
    ParseInputDataType(ctx.InputVar(name), name, &data_type);
  } else {
    ParseMultiInputDataType(ctx.MultiInputVar(name), name, &data_type);
  }
  PADDLE_ENFORCE_NE(
      data_type,
      default_data_type,
      common::errors::InvalidArgument(
          "The Input Variable(%s) of (%s) Operator used to determine kernel "
          "data type is empty or not phi::DenseTensor or SelectedRows or "
          "DenseTensorArray.",
          name,
          Type()));
  return data_type;
}

phi::DenseTensor* OperatorWithKernel::GetTensorFormInputSafely(
    const ExecutionContext& ctx, const std::string& name) const {
  // 1. get variable and check
  // NOTE: only supports signal input var now
  // NOTE: using const_cast is because we don't have method
  // can get single mutable var, and here will not change
  // the var's data, only use some attribute
  Variable* var = const_cast<Variable*>(ctx.InputVar(name));
  PADDLE_ENFORCE_NOT_NULL(
      var,
      common::errors::NotFound(
          "The variable %s is not found when promote complex types.", name));
  // 2. get tensor and check
  phi::DenseTensor* t = nullptr;
  if (var->IsType<phi::DenseTensor>()) {
    t = var->GetMutable<phi::DenseTensor>();
  } else if (var->IsType<phi::SelectedRows>()) {
    t = var->GetMutable<phi::SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported input variable type in complex type promotion."));
  }
  PADDLE_ENFORCE_NOT_NULL(t,
                          common::errors::InvalidArgument(
                              "The phi::DenseTensor of variable %s is nullptr "
                              "when promote complex types."));
  PADDLE_ENFORCE_EQ(
      t->IsInitialized(),
      true,
      common::errors::InvalidArgument(
          "The phi::DenseTensor in the %s Op's Input Variable %s(%s) is "
          "not initialized.",
          Type(),
          name,
          ctx.InputName(name)));
  return t;
}

/** NOTE(chenweihang): For safety reasons, we now only
 * perform type promotes for binary operations with
 * complex type inputs, which is used to support the
 * paddle quantum function.
 * In other cases, the first input data type is used as
 * the kernel data type.
 */
proto::VarType::Type OperatorWithKernel::IndicateOrPromoteVarDataTypes(
    const ExecutionContext& ctx,
    const std::string& name1,
    const std::string& name2) const {
  // 1. Get tensor
  auto* tensor_a = GetTensorFormInputSafely(ctx, name1);
  auto* tensor_b = GetTensorFormInputSafely(ctx, name2);

  // 2. Get two input types
  auto type_a = framework::TransToProtoVarType(tensor_a->dtype());
  auto type_b = framework::TransToProtoVarType(tensor_b->dtype());

  // 3. Get first input type or promote complex types
  auto target_type = PromoteTypesIfComplexExists(type_a, type_b);

  return target_type;
}

phi::KernelKey OperatorWithKernel::GetExpectedKernelType(
    const ExecutionContext& ctx) const {
  return phi::KernelKey(IndicateDataType(ctx), ctx.GetPlace());
}

phi::KernelKey OperatorWithKernel::GetKernelTypeForVar(
    const std::string& var_name,
    const phi::DenseTensor& tensor,
    const phi::KernelKey& expected_kernel_type) const {
#ifdef PADDLE_WITH_DNNL
  // When the op is first oneDNN op (there was some non oneDNN op
  // previously)
  // then we also need to rotate shape NHWC -> NCWH
  if ((expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
      (tensor.layout() != phi::DataLayout::ONEDNN) &&
      phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
          phi::DataLayout::kNHWC) {
    return phi::KernelKey(
        tensor.place(), phi::DataLayout::kNHWC, expected_kernel_type.dtype());
  }
#endif
  return phi::KernelKey(
      tensor.place(), tensor.layout(), expected_kernel_type.dtype());
}

phi::KernelSignature OperatorWithKernel::GetExpectedPhiKernelArgs(
    const ExecutionContext& ctx) const {
  ExecutionArgumentMappingContext arg_mapping_ctx(ctx);
  if (arg_map_fn_ == nullptr) {
    auto* arg_map_fn = phi::OpUtilsMap::Instance().GetArgumentMappingFn(type_);
    if (arg_map_fn) {
      arg_map_fn_ = std::make_unique<phi::ArgumentMappingFn>(*arg_map_fn);
    } else {
      auto func =
          [this](
              const phi::ArgumentMappingContext& ctx) -> phi::KernelSignature {
        return phi::DefaultKernelSignatureMap::Instance().Get(type_);
      };
      arg_map_fn_ = std::make_unique<phi::ArgumentMappingFn>(func);
    }
  }
  return (*arg_map_fn_)(arg_mapping_ctx);
}

static void SetDnnAttrIntoDeviceContext(
    phi::DeviceContext* dev_ctx,
    const Attribute& attr,
    const std::string& attr_name,
    const operators::ExtraAttrPropertySet& attr_properties) {
#ifdef PADDLE_WITH_DNNL
  if (phi::OneDNNContext::classof(dev_ctx) &&
      attr_properties.Support(operators::ExtraAttrProperty::ONEDNN)) {
    VLOG(4) << "Runtime attr `" << attr_name << "` is passed to OneDNNContext.";
    phi::OneDNNContext* one_dnn_ctx = static_cast<phi::OneDNNContext*>(dev_ctx);
    switch (AttrTypeID(attr)) {
      case proto::AttrType::FLOAT:
        one_dnn_ctx->SetDnnAttr(attr_name, PADDLE_GET_CONST(float, attr));
        break;
      case proto::AttrType::INT:
        one_dnn_ctx->SetDnnAttr(attr_name, PADDLE_GET_CONST(int, attr));
        break;
      case proto::AttrType::STRING:
        one_dnn_ctx->SetDnnAttr(attr_name, PADDLE_GET_CONST(std::string, attr));
        break;
      case proto::AttrType::INTS:  // NOLINT
        one_dnn_ctx->SetDnnAttr(attr_name,
                                PADDLE_GET_CONST(std::vector<int>, attr));
        break;
      case proto::AttrType::FLOATS:
        one_dnn_ctx->SetDnnAttr(attr_name,
                                PADDLE_GET_CONST(std::vector<float>, attr));
        break;
      case proto::AttrType::BOOLEAN:
        one_dnn_ctx->SetDnnAttr(attr_name, PADDLE_GET_CONST(bool, attr));
        break;
      default:
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported Attribute value type `%s` for phi.",
            common::demangle(attr.type().name())));
    }
  }
#endif
#ifdef PADDLE_WITH_CUDA
  if (phi::GPUContext::classof(dev_ctx) &&
      attr_properties.Support(operators::ExtraAttrProperty::GPUDNN)) {
    VLOG(4) << "Runtime attr `" << attr_name << "` is passed to GPUDNNContext.";
    phi::GPUContext* gpu_dnn_ctx = static_cast<phi::GPUContext*>(dev_ctx);
    switch (AttrTypeID(attr)) {
      case proto::AttrType::INT:
        gpu_dnn_ctx->SetDnnAttr(attr_name, PADDLE_GET_CONST(int, attr));
        break;
      case proto::AttrType::BOOLEAN:
        gpu_dnn_ctx->SetDnnAttr(attr_name, PADDLE_GET_CONST(bool, attr));
        break;
      default:
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported Attribute value type `%s` for phi.",
            common::demangle(attr.type().name())));
    }
  }
#endif
}

void OperatorWithKernel::BuildPhiKernelContext(
    const RuntimeContext& ctx,
    phi::DeviceContext* dev_ctx,
    phi::KernelContext* phi_kernel_context) const {
  phi_kernel_context->SetDeviceContext(dev_ctx);

  auto& input_names = kernel_signature_->input_names;
  auto& attr_names = kernel_signature_->attr_names;
  auto& output_names = kernel_signature_->output_names;

  auto input_defs = phi_kernel_->args_def().input_defs();
  auto attr_defs = phi_kernel_->args_def().attribute_defs();
  auto output_defs = phi_kernel_->args_def().output_defs();

#if defined(PADDLE_WITH_DNNL)
  if (phi::OneDNNContext::classof(dev_ctx)) {
    // Onednn holds this op's variable's name and init them here.
    phi::OneDNNContext* one_dnn_ctx = static_cast<phi::OneDNNContext*>(dev_ctx);
    one_dnn_ctx->SetInputsName(Inputs());
    one_dnn_ctx->SetOutputsName(Outputs());
  }
#endif

  PADDLE_ENFORCE_EQ(input_names.size(),
                    input_defs.size(),
                    common::errors::InvalidArgument(
                        "The size of inputs_args names (%d) must be equal to "
                        "the size of kernel input_defs (%d).",
                        input_names.size(),
                        input_defs.size()));

  PADDLE_ENFORCE_EQ(output_names.size(),
                    output_defs.size(),
                    common::errors::InvalidArgument(
                        "The size of outputs_args names (%d) must be equal to "
                        "the size of kernel output_defs (%d).",
                        output_names.size(),
                        output_defs.size()));

  PADDLE_ENFORCE_EQ(attr_names.size(),
                    attr_defs.size(),
                    common::errors::InvalidArgument(
                        "The size of attribute_args names (%d) must be equal "
                        "to the size of kernel attribute_defs (%d).",
                        attr_names.size(),
                        attr_defs.size()));
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto it = ctx.inputs.find(input_names[i]);

    // calculate the start and end index of the input tensors
    size_t start_idx =
        (i == 0 ? 0 : phi_kernel_context->InputRangeAt(i - 1).second);
    // deal with optional here
    if ((it == ctx.inputs.end() || it->second.empty()) &&
        (input_defs[i].type_index ==
             std::type_index(typeid(paddle::optional<phi::DenseTensor>)) ||
         input_defs[i].type_index ==
             std::type_index(typeid(paddle::optional<phi::SelectedRows>)) ||
         input_defs[i].type_index ==
             std::type_index(typeid(
                 paddle::optional<std::vector<const phi::DenseTensor*>>)))) {
      phi_kernel_context->EmplaceBackInputWithoutSetRange(nullptr);
      auto end_idx = start_idx + 1;
      phi_kernel_context->AssignInputRange(std::make_pair(start_idx, end_idx),
                                           i);

      continue;
    }
    auto ins_vector = it->second;
    size_t end_idx = start_idx + ins_vector.size();
    for (auto* var : ins_vector) {
      const phi::TensorBase* tensor_in = nullptr;
      if (var->IsType<phi::DenseTensor>()) {
        tensor_in = &(var->Get<phi::DenseTensor>());
        phi_kernel_context->EmplaceBackInputWithoutSetRange(tensor_in);
      } else if (var->IsType<phi::SelectedRows>()) {
        tensor_in = &(var->Get<phi::SelectedRows>());
        phi_kernel_context->EmplaceBackInputWithoutSetRange(tensor_in);
      } else if (var->IsType<phi::SparseCooTensor>()) {
        tensor_in = &(var->Get<phi::SparseCooTensor>());
        phi_kernel_context->EmplaceBackInputWithoutSetRange(tensor_in);
      } else if (var->IsType<phi::TensorArray>()) {
        need_prepare_phi_data_ = true;
        tensor_in = &(var->Get<phi::TensorArray>());
        phi_kernel_context->EmplaceBackInputWithoutSetRange(tensor_in);
      } else if (var->IsType<phi::Vocab>()) {
        tensor_in = &(var->Get<phi::Vocab>());
        phi_kernel_context->EmplaceBackInputWithoutSetRange(tensor_in);
      } else if (var->IsType<phi::Strings>()) {
        tensor_in = &(var->Get<phi::Strings>());
        phi_kernel_context->EmplaceBackInputWithoutSetRange(tensor_in);
      } else if (var->IsType<framework::FeedList>()) {
        tensor_in = &(var->Get<framework::FeedList>());
        phi_kernel_context->EmplaceBackInputWithoutSetRange(tensor_in);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported input `%s` type when call pt kernel.",
            framework::ToTypeName(var->Type())));
      }
    }
    // Note: here cannot deal with vector<phi::TensorArray> input
    phi_kernel_context->AssignInputRange(std::make_pair(start_idx, end_idx), i);
  }
  VLOG(4) << "Done inputs";
  for (size_t i = 0; i < output_names.size(); ++i) {
    auto it = ctx.outputs.find(output_names[i]);
    size_t start_idx =
        (i == 0 ? 0 : phi_kernel_context->OutputRangeAt(i - 1).second);

    if (it == ctx.outputs.end() || it->second.empty()) {
      VLOG(4) << "Output " << output_names[i] << " not found";
      // Deal with the case that some outputs are not found or be NULL when run
      // the kernel.
      // For example : the outputs of matmul_grad are dx and dy,
      // sometimes dx or dy may be NULL.
      phi_kernel_context->EmplaceBackOutputWithoutSetRange(nullptr);
      auto end_idx = start_idx + 1;
      phi_kernel_context->AssignOutputRange(std::make_pair(start_idx, end_idx),
                                            i);
      continue;
    }
    auto& outs_vector = it->second;

    size_t end_idx = start_idx + outs_vector.size();

    for (auto* var : outs_vector) {
      phi::TensorBase* tensor_out = nullptr;
      if (var) {
        if (var->template IsType<phi::DenseTensor>()) {
          tensor_out = var->template GetMutable<phi::DenseTensor>();
          phi_kernel_context->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else if (var->template IsType<phi::SelectedRows>()) {
          tensor_out = var->template GetMutable<phi::SelectedRows>();
          phi_kernel_context->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else if (var->template IsType<phi::SparseCooTensor>()) {
          tensor_out = var->template GetMutable<phi::SparseCooTensor>();
          phi_kernel_context->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else if (var->template IsType<phi::TensorArray>()) {
          tensor_out = var->template GetMutable<phi::TensorArray>();
          // Note: If the input phi::TensorArray size is 0, the output
          // phi::TensorArray is also 0
          phi_kernel_context->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else if (var->template IsType<phi::Strings>()) {
          tensor_out = var->template GetMutable<phi::Strings>();
          phi_kernel_context->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else if (var->template IsType<phi::Vocab>()) {
          tensor_out = var->template GetMutable<phi::Vocab>();
          phi_kernel_context->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else if (var->template IsType<phi::RawTensor>() ||
                   !var->IsInitialized()) {
          tensor_out = var->template GetMutable<phi::RawTensor>();
          phi_kernel_context->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "Unsupported output `%s` type when call pt kernel.",
              framework::ToTypeName(var->Type())));
        }
      } else {
        VLOG(4) << "Output " << output_names[i] << " is nullptr";
        phi_kernel_context->EmplaceBackOutputWithoutSetRange(tensor_out);
      }
    }
    phi_kernel_context->AssignOutputRange(std::make_pair(start_idx, end_idx),
                                          i);
  }
  VLOG(4) << "Done outputs";
  for (size_t i = 0; i < attr_names.size(); ++i) {
    VLOG(6) << "BuildPhiKernelContext: " << attr_names[i] << ": "
            << attr_defs[i].type_index;
    // attribute with Variable type has been placed into Inputs(), and
    // we can parse them from RuntimeContext.inputs.
    auto attr_iter = Attrs().find(attr_names[i]);
    switch (attr_defs[i].type_index) {
      case phi::AttributeType::SCALAR:
        if (attr_iter != Attrs().end()) {
          // scalar is in the attribute
          switch (AttrTypeID(attr_iter->second)) {
            case proto::AttrType::FLOAT:
              phi_kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(float, attr_iter->second)));
              break;
            case proto::AttrType::FLOAT64:
              phi_kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(double, attr_iter->second)));
              break;
            case proto::AttrType::INT:
              phi_kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(int, attr_iter->second)));
              break;
            case proto::AttrType::LONG:
              phi_kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(int64_t, attr_iter->second)));
              break;
            case proto::AttrType::STRING:
              phi_kernel_context->EmplaceBackAttr(phi::Scalar(
                  PADDLE_GET_CONST(std::string, attr_iter->second)));
              break;
            case proto::AttrType::BOOLEAN:
              phi_kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(bool, attr_iter->second)));
              break;
            case proto::AttrType::SCALAR:
              phi_kernel_context->EmplaceBackAttr(phi::Scalar(PADDLE_GET_CONST(
                  paddle::experimental::Scalar, attr_iter->second)));
              break;
            default:
              PADDLE_THROW(common::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to Scalar when construct "
                  "KernelContext in dygraph.",
                  attr_names[i]));
          }
        } else {  // scalar is in the input
          need_prepare_phi_data_ = true;
          auto& ins_vector = ctx.inputs.at(attr_names[i]);
          phi_kernel_context->EmplaceBackAttr(
              framework::MakePhiScalarFromVar(*ins_vector.front()));
        }
        break;
      case phi::AttributeType::INT_ARRAY:
        if (attr_iter != Attrs().end()) {
          switch (AttrTypeID(attr_iter->second)) {
            case proto::AttrType::INTS:  // NOLINT
              phi_kernel_context->EmplaceBackAttr(phi::IntArray(
                  PADDLE_GET_CONST(std::vector<int32_t>, attr_iter->second)));
              break;
            case proto::AttrType::LONGS:
              phi_kernel_context->EmplaceBackAttr(phi::IntArray(
                  PADDLE_GET_CONST(std::vector<int64_t>, attr_iter->second)));
              break;
            case proto::AttrType::INT:
              phi_kernel_context->EmplaceBackAttr(phi::IntArray(
                  &PADDLE_GET_CONST(int32_t, attr_iter->second), 1));
              break;
            case proto::AttrType::LONG:
              phi_kernel_context->EmplaceBackAttr(phi::IntArray(
                  &PADDLE_GET_CONST(int64_t, attr_iter->second), 1));
              break;
            default:
              PADDLE_THROW(common::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to IntArray when "
                  "construct KernelContext.",
                  attr_names[i]));
          }
        } else {  // shape is in the input
          need_prepare_phi_data_ = true;
          auto& ins_vector = ctx.inputs.at(attr_names[i]);
          if (ins_vector.size() == 1) {  // ShapeTensor
            phi_kernel_context->EmplaceBackAttr(
                framework::MakePhiIntArrayFromVar(*ins_vector.front()));
          } else {  // ShapeTensorList
            phi_kernel_context->EmplaceBackAttr(
                framework::MakePhiIntArrayFromVarList(ins_vector));
          }
        }
        break;

      case phi::AttributeType::SCALARS: {
        PADDLE_ENFORCE_NE(
            attr_iter,
            Attrs().end(),
            common::errors::NotFound("(%s) is not found in AttributeMap when "
                                     "building static KernelContext.",
                                     attr_names[i]));
        switch (AttrTypeID(attr_iter->second)) {
          case proto::AttrType::INTS: {
            const auto& vec =
                PADDLE_GET_CONST(std::vector<int32_t>, attr_iter->second);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            phi_kernel_context->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case proto::AttrType::LONGS: {
            const auto& vec =
                PADDLE_GET_CONST(std::vector<int64_t>, attr_iter->second);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            phi_kernel_context->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case proto::AttrType::FLOATS: {
            const auto& vec =
                PADDLE_GET_CONST(std::vector<float>, attr_iter->second);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            phi_kernel_context->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case proto::AttrType::FLOAT64S: {
            const auto& vec =
                PADDLE_GET_CONST(std::vector<double>, attr_iter->second);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            phi_kernel_context->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case proto::AttrType::BOOLEANS: {
            const auto& vec =
                PADDLE_GET_CONST(std::vector<bool>, attr_iter->second);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            phi_kernel_context->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case proto::AttrType::SCALARS: {
            const auto& vec = PADDLE_GET_CONST(
                std::vector<paddle::experimental::Scalar>, attr_iter->second);
            std::vector<phi::Scalar> scalar_list{vec.begin(), vec.end()};
            phi_kernel_context->EmplaceBackAttr(std::move(scalar_list));
          } break;
          default:
            PADDLE_THROW(common::errors::Unimplemented(
                "Unsupported cast op attribute `%s` to vector<Scalar> when "
                "construct KernelContext.",
                attr_names[i]));
        }
      } break;
      default: {
        if (attr_iter == Attrs().end()) {
          // TODO(chenweihang): remove this backup searching later
          attr_iter = RuntimeAttrs().find(attr_names[i]);
          PADDLE_ENFORCE_NE(
              attr_iter,
              RuntimeAttrs().end(),
              common::errors::NotFound("(%s) is not found in AttributeMap when "
                                       "building static KernelContext.",
                                       attr_names[i]));
        }

        switch (attr_defs[i].type_index) {
          case phi::AttributeType::FLOAT32:
            phi_kernel_context->EmplaceBackAttr(
                PADDLE_GET_CONST(float, attr_iter->second));
            break;
          case phi::AttributeType::FLOAT64:
            phi_kernel_context->EmplaceBackAttr(
                PADDLE_GET_CONST(double, attr_iter->second));
            break;
          case phi::AttributeType::INT32:
            phi_kernel_context->EmplaceBackAttr(
                PADDLE_GET_CONST(int, attr_iter->second));
            break;
          case phi::AttributeType::BOOL:
            phi_kernel_context->EmplaceBackAttr(
                PADDLE_GET_CONST(bool, attr_iter->second));
            break;
          case phi::AttributeType::INT64:
            phi_kernel_context->EmplaceBackAttr(
                PADDLE_GET_CONST(int64_t, attr_iter->second));
            break;
          case phi::AttributeType::INT32S:  // NOLINT
            phi_kernel_context->EmplaceBackAttr(
                PADDLE_GET_CONST(std::vector<int>, attr_iter->second));
            break;
          case phi::AttributeType::BOOLS:
            phi_kernel_context->EmplaceBackAttr(
                PADDLE_GET_CONST(std::vector<bool>, attr_iter->second));
            break;
          case phi::AttributeType::DATA_TYPE: {
            auto data_type = phi::TransToPhiDataType(
                static_cast<framework::proto::VarType::Type>(
                    PADDLE_GET_CONST(int, attr_iter->second)));
            phi_kernel_context->EmplaceBackAttr(data_type);
          } break;
          case phi::AttributeType::STRING:
            phi_kernel_context->EmplaceBackAttr(
                PADDLE_GET_CONST(std::string, attr_iter->second));
            break;
          case phi::AttributeType::INT64S:
            switch (AttrTypeID(attr_iter->second)) {
              case proto::AttrType::LONGS:
                phi_kernel_context->EmplaceBackAttr(
                    PADDLE_GET_CONST(std::vector<int64_t>, attr_iter->second));
                break;
              case proto::AttrType::INTS: {
                const auto& vector_int_attr =
                    PADDLE_GET_CONST(std::vector<int>, attr_iter->second);
                const std::vector<int64_t> vector_int64_attr(
                    vector_int_attr.begin(), vector_int_attr.end());
                phi_kernel_context->EmplaceBackAttr(vector_int64_attr);
              } break;
              default:
                PADDLE_THROW(common::errors::Unimplemented(
                    "Unsupported cast op attribute `%s` to vector<int64_t> "
                    "when "
                    "construct KernelContext.",
                    attr_names[i]));
            }
            break;
          case phi::AttributeType::FLOAT32S:  // NOLINT
            phi_kernel_context->EmplaceBackAttr(
                PADDLE_GET_CONST(std::vector<float>, attr_iter->second));
            break;
          case phi::AttributeType::STRINGS:
            phi_kernel_context->EmplaceBackAttr(
                PADDLE_GET_CONST(std::vector<std::string>, attr_iter->second));
            break;
          default:
            PADDLE_THROW(common::errors::Unimplemented(
                "Unsupported cast op attribute `%s` when construct "
                "KernelContext in dygraph.",
                attr_names[i]));
        }
      }
    }
  }
  VLOG(4) << "Done attributes";

// Clear All old attrs before add new attrs,
// because sometimes old attrs may be misused.
#if defined(PADDLE_WITH_DNNL)
  if (phi::OneDNNContext::classof(dev_ctx)) {
    phi::OneDNNContext* one_dnn_ctx = static_cast<phi::OneDNNContext*>(dev_ctx);
    one_dnn_ctx->ClearDnnAttr();
    if (!RuntimeAttrs().empty()) need_prepare_phi_data_ = true;
  }
#endif

  // Note(YuanRisheng): Now, we can't open code below.
  // Because some unittest run OLD dygraph and ExtraAttr is not supported in OLD
  // dygraph. So, here we use trick that dev_ctx is a global object. We can
  // store ExtraAttr in static graph and when unittest run OLD dygraph, it can
  // obtain these ExtraAttr. We can open this code when OLD dygraph is no longer
  // used.
  /*
  #if defined(PADDLE_WITH_CUDA)
    if(phi::GPUContext::classof(dev_ctx)) {
      phi::GPUContext* gpu_dnn_ctx = static_cast<phi::GPUContext*>(dev_ctx);
      gpu_dnn_ctx->ClearDnnAttr();
    }
  #endif
  */
  // For compatible with Op with extra attrs for specific backend
#if defined(PADDLE_WITH_DNNL) || defined(PADDLE_WITH_CUDA)
  auto& runtime_attrs = RuntimeAttrs();
  for (const auto& attr_iter : runtime_attrs) {
    auto& attr_name = attr_iter.first;
    auto& attr = attr_iter.second;
    auto attr_properties = paddle::operators::GetExtraAttrProperties(attr_name);
    SetDnnAttrIntoDeviceContext(dev_ctx, attr, attr_name, attr_properties);
  }
  // TODO(chenweihang): Since the pass will still `SetAttr` in the OpDesc,
  // we try to add these Attrs to the RuntimeAttrs, but these OpDesc will lose
  // the RuntimeAttrs information in the process of converting the Graph to
  // the Program, so additional record configuration will be introduced,
  // which increases the cost of development and understanding, so we
  // still use Attrs to get and the attributes set by these passes from Attrs
  // for the time being. In the future, it is necessary to clarify the
  // positioning of RuntimeAttrs and expand related functions.
  auto& attrs = Attrs();
  for (const auto& attr_iter : attrs) {
    auto& attr_name = attr_iter.first;
    auto& attr = attr_iter.second;
    auto attr_properties = paddle::operators::GetExtraAttrProperties(attr_name);
    SetDnnAttrIntoDeviceContext(dev_ctx, attr, attr_name, attr_properties);
  }
  VLOG(4) << "Done runtime attributes";
#endif

// For compatible with Op with extra input for onednn backend
#ifdef PADDLE_WITH_DNNL
  if (phi::OneDNNContext::classof(dev_ctx)) {
    phi::OneDNNContext* one_dnn_ctx = static_cast<phi::OneDNNContext*>(dev_ctx);
    auto& extra_input_names =
        paddle::operators::ExtraInfoUtils::Instance().GetExtraInputNamesMap(
            Type());
    for (const auto& input_name : extra_input_names) {
      auto it = ctx.inputs.find(input_name);
      if (it == ctx.inputs.end() || it->second.empty()) {
        one_dnn_ctx->SetDnnInput(input_name, nullptr);
      } else {
        auto ins_vector = it->second;
        PADDLE_ENFORCE_EQ(
            ins_vector.size(),
            1UL,
            common::errors::InvalidArgument(
                "OneDNN's extra input only allows one input tensor."));
        auto* var = ins_vector[0];
        PADDLE_ENFORCE_EQ(var->IsType<phi::DenseTensor>(),
                          true,
                          common::errors::InvalidArgument(
                              "OneDNN's extra input only can be DenseTensor."));
        one_dnn_ctx->SetDnnInput(input_name, &(var->Get<phi::DenseTensor>()));
      }
    }
  }
  VLOG(4) << "Done runtime extra inputs";
#endif
}

}  // namespace paddle::framework
