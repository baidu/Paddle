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

#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"  // For VLOG
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/unused_var_check.h"
#include "paddle/phi/core/memory/malloc.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/vocab/string_array.h"

#include "paddle/common/flags.h"
#include "paddle/common/macros.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace framework {
class OpInfo;
class Scope;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace phi {
class KernelContext;
}

COMMON_DECLARE_int32(inner_op_parallelism);

namespace paddle {
namespace framework {

constexpr char kFakeVarName[] = "Fake_var";

/// If a variable is a empty variable, that name will be used.
constexpr char kEmptyVarName[] = "@EMPTY@";

/// If a variable is a temporary variable, that name will be set in Python,
/// but it will be convert to a unique name in scope after OpCreator.
constexpr char kTempVarName[] = "@TEMP@";

/// If a variable's name has a certain suffix, it means that the
/// variable is the gradient of another variable.
/// e.g. Variable "x@GRAD" is the gradient of variable "x".
constexpr char kGradVarSuffix[] = "@GRAD";

constexpr size_t kGradVarSuffixSize = 5U;

/// Variables with this suffix are supposed to be filled up with zeros.
constexpr char kZeroVarSuffix[] = "@ZERO";

/// Variables with this suffix are the new Gradient.
constexpr char kNewGradSuffix[] = "@NEWGRAD@";

/// RuntimeContext is used to relate input/output names of Operator with
/// the corresponding variables in name scope.
/// If an Op has attribute kEnableCacheRuntimeContext, it means that in a same
/// name scope, since the input/output names of this Op do not change in the
/// execution, RuntimeContext could be created only at the first iteration of
/// this Op's execution to save the elapsed time.
constexpr char kEnableCacheRuntimeContext[] = "@ENABLE_CACHE_RUNTIME_CONTEXT@";

/// If an Op has this attribute, all its kernels should calculate output
/// variable's shape in the corresponding Compute() function. And
/// OperatorWithKernel::RunImpl() would skip call this Op's InferShape()
/// function in its runtime for speedup.
/// TODO(luotao): Note that this temporal attribute would be deleted after all
/// ops contain it.
constexpr char kAllKernelsMustComputeRuntimeShape[] =
    "ALL_KERNELS_MUST_COMPUTE_RUNTIME_SHAPE";

// define some kernel priority
/* Define multiple kernel type fallback order*/
extern std::vector<std::tuple<phi::Place, LibraryType>> kKernelPriority;

inline std::string GradVarName(const std::string& var_name) {
  std::string result;
  result.reserve(var_name.size() + kGradVarSuffixSize);
  result += var_name;
  result += kGradVarSuffix;
  return result;
}

inline std::string GradOriginalVarName(const std::string& grad_var_name) {
  std::size_t pos = grad_var_name.rfind(kGradVarSuffix);
  if (pos == std::string::npos) {
    return grad_var_name;
  } else {
    return grad_var_name.substr(0, pos);
  }
}

inline bool VarIsTensor(const Variable& var) {
  return var.IsType<phi::DenseTensor>() || var.IsType<phi::SelectedRows>();
}

const phi::DenseTensor* GetDenseTensorOrSelectedRowsValueFromVar(
    const Variable& var);
phi::DenseTensor* GetMutableDenseTensorOrSelectedRowsValueFromVar(
    Variable* var);

class ExecutionContext;
class OperatorBase;

class RuntimeContext {
 public:
  RuntimeContext(const VariableNameMap& innames,
                 const VariableNameMap& outnames,
                 const Scope& scope);

  RuntimeContext(const VariableValueMap& invars,
                 const VariableValueMap& outvars)
      : inputs(invars), outputs(outvars) {}

  VariableValueMap inputs;
  VariableValueMap outputs;
};

class RuntimeInferShapeContext : public InferShapeContext {
 public:
  RuntimeInferShapeContext(const OperatorBase& op, const RuntimeContext& ctx);

  bool HasInput(const std::string& name) const override;

  bool HasOutput(const std::string& name) const override;

  bool HasAttr(const std::string& name) const override;

  bool HasInputs(const std::string& name) const override;

  bool HasOutputs(const std::string& name,
                  bool allow_null = false) const override;

  AttrReader Attrs() const override;

  std::vector<std::string> Inputs(const std::string& name) const override;

  std::vector<std::string> Outputs(const std::string& name) const override;

  std::string GetInputNameByIdx(size_t idx) const override;

  std::string GetOutputNameByIdx(size_t idx) const override;

  void ShareDim(const std::string& in,
                const std::string& out,
                size_t i = 0,
                size_t j = 0) override;

  void ShareAllLoD(const std::string& in,
                   const std::string& out) const override;

  void ShareLoD(const std::string& in,
                const std::string& out,
                size_t i = 0,
                size_t j = 0) const override;

  int32_t GetLoDLevel(const std::string& in, size_t i = 0) const override;

  void SetLoDLevel(const std::string& out,
                   int32_t lod_level,
                   size_t j = 0) const override;

  bool IsRuntime() const override;

  bool IsRunMKLDNNKernel() const override;

  // TODO(paddle-dev): Can this be template?
  paddle::small_vector<InferShapeVarPtr, phi::kInputSmallVectorSize>
  GetInputVarPtrs(const std::string& name) const override;

  paddle::small_vector<InferShapeVarPtr, phi::kOutputSmallVectorSize>
  GetOutputVarPtrs(const std::string& name) const override;

  DDim GetInputDim(const std::string& name) const override;

  std::vector<DDim> GetInputsDim(const std::string& name) const override;

  proto::VarType::Type GetInputVarType(const std::string& name) const override;

  std::vector<proto::VarType::Type> GetInputsVarType(
      const std::string& name) const override;

  std::vector<proto::VarType::Type> GetOutputsVarType(
      const std::string& name) const override;

  void SetOutputDim(const std::string& name, const DDim& dim) override;

  void SetOutputsDim(const std::string& name,
                     const std::vector<DDim>& dims) override;

  const phi::ArgumentMappingFn* GetPhiArgumentMappingFn() const override;

  const phi::KernelSignature* GetPhiDefaultKernelSignature() const override;

  void SetSkipLoD(bool skip);

  std::vector<LegacyLoD> GetOutputsLod(const std::string& out) const;

  std::vector<DDim> GetOutputsDim(const std::string& name) const;

  bool HasRuntimeAttributes() const;

 protected:
  DDim GetDim(Variable* var) const;

  std::vector<DDim> GetDims(const std::vector<Variable*>& vars) const;

  std::vector<DDim> GetRepeatedDims(const std::string& name) const override;

  void SetDim(Variable* var, const DDim& dim);

  void SetDims(const std::vector<Variable*>& vars,
               const std::vector<DDim>& dims);

  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override;

  std::vector<proto::VarType::Type> GetVarTypes(
      const std::vector<Variable*>& vars) const;

  proto::VarType::Type GetVarType(Variable* var) const;

 private:
  const std::vector<Variable*>& InputVars(const std::string& name) const;

  const std::vector<Variable*>& OutputVars(const std::string& name) const;

  const OperatorBase& op_;
  const RuntimeContext& ctx_;
  bool can_skip_lod_{false};
};

/**
 * OperatorBase has the basic elements that Net will call to do computation.
 * Only CreateOperator from OpRegistry will new Operator directly. User
 * should always construct a proto message OpDesc and call
 * OpRegistry::CreateOp(op_desc) to get an Operator instance.
 */
class TEST_API OperatorBase {
 public:
  OperatorBase(const std::string& type,
               const VariableNameMap& inputs,
               const VariableNameMap& outputs,
               const AttributeMap& attrs);

  virtual ~OperatorBase() {}

  /// Executor will call this interface function to Run an op.
  //  The implementation should be written at RunImpl
  void Run(const Scope& scope, const phi::Place& place);

  // FIXME(typhoonzero): this is only used for recv_op to stop event_loop.
  virtual void Stop() {}

  /// if scope is not null, also show dimensions of arguments
  virtual std::string DebugStringEx(const Scope* scope) const;
  std::string DebugString() const { return DebugStringEx(nullptr); }

  virtual bool SupportGPU() const { return false; }
  virtual bool SupportXPU() const { return false; }
  virtual bool SupportCustomDevice() const { return false; }

  const std::string& Type() const { return type_; }

  bool HasAttr(const std::string& name) const {
    return attrs_.count(name) || runtime_attrs_.count(name);
  }
  template <typename T>
  inline const T& Attr(const std::string& name) const {
    auto it = attrs_.find(name);
    if (it == attrs_.end()) {
      it = runtime_attrs_.find(name);
      PADDLE_ENFORCE_NE(
          it,
          runtime_attrs_.end(),
          common::errors::NotFound(
              "(%s) is not found in AttributeMap and RuntimeAttributeMap.",
              name));
    }
    return PADDLE_GET_CONST(T, it->second);
  }
  void SetAttr(const std::string& name, const Attribute& v) {
    PADDLE_ENFORCE_EQ(
        HasAttr(name),
        true,
        common::errors::NotFound(
            "The attribute %s is not found in operator %s", name, Type()));

    attrs_[name] = v;
  }
  const AttributeMap& Attrs() const { return attrs_; }
  const AttributeMap& RuntimeAttrs() const { return runtime_attrs_; }
  void SetRuntimeAttributeMap(const AttributeMap& runtime_attrs) {
    runtime_attrs_ = runtime_attrs;
  }

  const VariableNameMap& Inputs() const { return inputs_; }
  const VariableNameMap& Outputs() const { return outputs_; }
  VariableNameMap& Inputs() { return inputs_; }
  VariableNameMap& Outputs() { return outputs_; }

  const OpInfo& Info() const {
    PADDLE_ENFORCE_NOT_NULL(
        info_,
        common::errors::NotFound("OpInfo of operator (%s) is not found.",
                                 type_));
    return *info_;
  }

  bool HasInputs(const std::string& name) const;
  //! Get a input with argument's name described in `op_proto`
  std::string Input(const std::string& name) const;
  //! Get a input which has multiple variables.
  const std::vector<std::string>& Inputs(const std::string& name) const;
  //! Get all inputs variable names
  std::vector<std::string> InputVars() const;

  bool HasOutputs(const std::string& name) const;
  //! Get a output with argument's name described in `op_proto`
  std::string Output(const std::string& name) const;
  //! Get an output which has multiple variables.
  //! TODO add a vector_view to prevent memory copy.
  const std::vector<std::string>& Outputs(const std::string& name) const;
  //! Get all outputs variable names
  virtual std::vector<std::string> OutputVars(bool has_intermediate) const;

  void SetIsCalledByExecutor(bool x) { run_by_executor_ = x; }

  virtual void SetIsRuntimeInferShape(bool x UNUSED) {}

  virtual void RuntimeInferShape(const Scope& scope UNUSED,
                                 const phi::Place& place UNUSED,
                                 const RuntimeContext& ctx UNUSED) const {}

  virtual phi::Place GetExecutionPlace(const phi::Place& place) const {
    return place;
  }

  uint64_t Id() const { return id_; }

  void SetId(uint64_t id) { id_ = id; }

  using HookFunc = std::function<void(OperatorBase*, Scope*)>;
  void SetOutputHooks(const std::vector<HookFunc>& hookfuncs) {
    output_hookfuncs_ = hookfuncs;
  }
  void SetInputHooks(const std::vector<HookFunc>& hookfuncs) {
    input_hookfuncs_ = hookfuncs;
  }

 protected:
  std::string type_;
  // NOTE: in case of OpGrad, inputs_ contains:
  // I (Inputs)
  // O (Outputs)
  // OG (Output Gradients)
  VariableNameMap inputs_;

  // NOTE: in case of OpGrad, outputs_ contains
  // IG (Inputs Gradients)
  VariableNameMap outputs_;
  AttributeMap attrs_;
  // NOTE: runtime_attrs_ contains the attributes which used for dispatching
  // kernel (use_mkldnn, use_cudnn, ...) or passing additional configuration
  // for special heterogeneous kernel (workspace_size_MB, ...).
  // The attributes in runtime_attrs_ are setted by framework (such as PASS),
  // and not in the python api.
  AttributeMap runtime_attrs_;

  // OpInfo
  const OpInfo* info_;

  // OpDesc Id
  uint64_t id_ = UINT64_MAX;

  // Whether this operator executes in an Executor.
  bool run_by_executor_{true};

  std::vector<HookFunc> output_hookfuncs_;
  std::vector<HookFunc> input_hookfuncs_;

 private:
  void GenerateTemporaryNames();
  void CheckAllInputOutputSet() const;
  virtual void RunImpl(const Scope& scope, const phi::Place& place) const = 0;
};

class ExecutionContext : public phi::KernelContext {
 public:
  ExecutionContext(const OperatorBase& op,
                   const Scope& scope,
                   const phi::DeviceContext& device_context,
                   const RuntimeContext& ctx)
      : op_(op), scope_(scope), device_context_(device_context), ctx_(ctx) {}
  virtual ~ExecutionContext() {}

  virtual std::string InputName(const std::string& name) const {
    return op_.Input(name);
  }
  virtual std::vector<std::string> InputNames(const std::string& name) const {
    return op_.Inputs(name);
  }
  virtual std::string OutputName(const std::string& name) const {
    return op_.Output(name);
  }

  virtual std::vector<std::string> OutputNames(const std::string& name) const {
    return op_.Outputs(name);
  }

  virtual bool HasAttr(const std::string& name) const {
    return op_.HasAttr(name);
  }
  virtual const AttributeMap& Attrs() const { return op_.Attrs(); }

  const std::string& Type() const { return op_.Type(); }

  const Scope& scope() const { return scope_; }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return PADDLE_GET_CONST(T, GetAttr(name));
  }

  virtual const Attribute& GetAttr(const std::string& name) const {
    auto iter = op_.Attrs().find(name);
    if (iter == op_.Attrs().end()) {
      iter = op_.RuntimeAttrs().find(name);
      PADDLE_ENFORCE_NE(
          iter,
          op_.RuntimeAttrs().end(),
          common::errors::NotFound("(%s) is not found in AttributeMap and "
                                   "RuntimeAttributeMap of (%s) operator.",
                                   name,
                                   op_.Type()));
    }
    return iter->second;
  }

  virtual bool HasInput(const std::string& name) const;

  virtual bool HasInputs(const std::string& name) const;

  virtual bool HasOutput(const std::string& name) const;

  virtual size_t InputSize(const std::string& name) const {
    return op_.Inputs(name).size();
  }

  virtual size_t OutputSize(const std::string& name) const {
    return op_.Outputs(name).size();
  }

  virtual const Variable* InputVar(const std::string& name) const;

  virtual Variable* OutputVar(const std::string& name) const;

  virtual const std::vector<Variable*> MultiInputVar(
      const std::string& name) const {
    LogVarUsageIfUnusedVarCheckEnabled(name);

    auto it = ctx_.inputs.find(name);
    if (it == ctx_.inputs.end()) {
      return {};
    }
    return {it->second.begin(), it->second.end()};
  }

  virtual std::vector<Variable*> MultiOutputVar(const std::string& name) const {
    auto it = ctx_.outputs.find(name);
    if (it == ctx_.outputs.end()) {
      return {};
    }
    return it->second;
  }

  virtual paddle::small_vector<const std::string*> InNameList() const {
    paddle::small_vector<const std::string*> vec_temp;
    vec_temp.reserve(ctx_.inputs.size());

    for (auto& input : ctx_.inputs) {
      vec_temp.push_back(&input.first);
    }

    return vec_temp;
  }

  template <typename T>
  const T* Input(const std::string& name) const {
    auto* var = InputVar(name);
    return var == nullptr ? nullptr : &var->Get<T>();
  }

  template <typename T>
  T* Output(const std::string& name) const {
    auto var = OutputVar(name);
    return var == nullptr ? nullptr : var->GetMutable<T>();
  }

  template <typename T>
  const std::vector<const T*> MultiInput(const std::string& name) const {
    LogVarUsageIfUnusedVarCheckEnabled(name);

    auto vars = MultiInputVar(name);
    if (vars.size() == 0) {
      return {};
    }
    std::vector<const T*> res;
    res.reserve(vars.size());
    std::transform(vars.begin(),
                   vars.end(),
                   std::back_inserter(res),
                   [&](const Variable* var) -> const T* {
                     return var == nullptr ? nullptr : &var->Get<T>();
                   });
    return res;
  }

  template <typename T>
  std::vector<T*> MultiOutput(const std::string& name) const {
    auto vars = MultiOutputVar(name);

    if (vars.size() == 0) {
      return {};
    }

    std::vector<T*> res;
    res.reserve(vars.size());
    std::transform(vars.begin(),
                   vars.end(),
                   std::back_inserter(res),
                   [&](Variable* var) -> T* {
                     return var == nullptr ? nullptr : var->GetMutable<T>();
                   });

    return res;
  }

  phi::Place GetPlace() const { return device_context_.GetPlace(); }

  template <typename DeviceContextType>
  const DeviceContextType& device_context() const {
    return *reinterpret_cast<const DeviceContextType*>(&device_context_);
  }

  const phi::DeviceContext& device_context() const { return device_context_; }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  const inline phi::GPUContext& cuda_device_context() const {
    PADDLE_ENFORCE_EQ(phi::is_gpu_place(device_context_.GetPlace()),
                      true,
                      common::errors::PreconditionNotMet(
                          "Current device context place is not GPUPlace."));
    return *reinterpret_cast<const phi::GPUContext*>(&device_context_);
  }
#endif

  template <typename T, typename DevContext>
  phi::DenseTensor AllocateTmpTensor(const phi::DDim& dim,
                                     const DevContext& dev_ctx) const {
    phi::DenseTensor tmp;
    tmp.Resize(dim);
    dev_ctx.template Alloc<T>(&tmp);
    return tmp;
  }

  const RuntimeContext Context() const { return ctx_; }

  std::string DebugString() const { return op_.DebugString(); }
  const OperatorBase& GetOp() const { return op_; }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
  const phi::DeviceContext& device_context_;
  const RuntimeContext& ctx_;
};

// TODO(chenweihang): split impl based OpProto or Dygraph if needed
class ExecutionArgumentMappingContext : public phi::ArgumentMappingContext {
 public:
  explicit ExecutionArgumentMappingContext(const ExecutionContext& ctx)
      : ctx_(ctx) {}

  bool HasInput(const std::string& name) const override {
    return ctx_.HasInputs(name);
  }

  bool HasOutput(const std::string& name) const override {
    return ctx_.HasOutput(name);
  }

  bool HasAttr(const std::string& name) const override {
    return ctx_.HasAttr(name);
  }

  paddle::any Attr(const std::string& name) const override {
    auto& attr = ctx_.GetAttr(name);
    return GetAttrValue(attr);
  }

  size_t InputSize(const std::string& name) const override {
    return ctx_.MultiInputVar(name).size();
  }

  size_t OutputSize(const std::string& name) const override {
    return ctx_.MultiOutputVar(name).size();
  }

  bool IsDenseTensorInput(const std::string& name) const override {
    const auto* var = ctx_.InputVar(name);
    return var->IsType<phi::DenseTensor>();
  }

  bool IsDenseTensorInputs(const std::string& name) const override {
    auto vars = ctx_.MultiInputVar(name);
    return std::all_of(vars.begin(), vars.end(), [](const Variable* var) {
      return var->IsType<phi::DenseTensor>();
    });
  }

  bool IsSelectedRowsInputs(const std::string& name) const override {
    auto vars = ctx_.MultiInputVar(name);
    return std::all_of(vars.begin(), vars.end(), [](const Variable* var) {
      return var->IsType<phi::SelectedRows>();
    });
  }

  bool IsSelectedRowsInput(const std::string& name) const override {
    const auto* var = ctx_.InputVar(name);
    return var->IsType<phi::SelectedRows>();
  }

  bool IsDenseTensorVectorInput(const std::string& name) const override {
    auto vars = ctx_.MultiInputVar(name);
    return std::all_of(vars.begin(), vars.end(), [](const Variable* var) {
      return var->IsType<phi::TensorArray>();
    });
  }

  bool IsSparseCooTensorInput(const std::string& name) const override {
    const auto* var = ctx_.InputVar(name);
    return var->IsType<phi::SparseCooTensor>();
  }

  bool IsSparseCooTensorOutput(const std::string& name) const override {
    auto vars = ctx_.MultiOutputVar(name);
    return std::all_of(vars.begin(), vars.end(), [](const Variable* var) {
      return var->IsType<phi::SparseCooTensor>();
    });
  }

  bool IsSparseCsrTensorInput(const std::string& name) const override {
    const auto* var = ctx_.InputVar(name);
    return var->IsType<phi::SparseCsrTensor>();
  }

  bool IsDenseTensorOutput(const std::string& name) const override {
    auto vars = ctx_.MultiOutputVar(name);
    return std::all_of(vars.begin(), vars.end(), [](const Variable* var) {
      return var->IsType<phi::DenseTensor>();
    });
  }

  bool IsVocabOutput(const std::string& name) const override {
    auto vars = ctx_.MultiOutputVar(name);
    return std::all_of(vars.begin(), vars.end(), [](const Variable* var) {
      return var->IsType<phi::Vocab>();
    });
  }

  bool IsSelectedRowsOutput(const std::string& name) const override {
    auto vars = ctx_.MultiOutputVar(name);
    return std::all_of(vars.begin(), vars.end(), [](const Variable* var) {
      return var->IsType<phi::SelectedRows>();
    });
  }

  bool IsForInferShape() const override { return false; }

 private:
  const ExecutionContext& ctx_;
};

template <>
const std::vector<const phi::DenseTensor*>
ExecutionContext::MultiInput<phi::DenseTensor>(const std::string& name) const;

template <>
std::vector<phi::DenseTensor*> ExecutionContext::MultiOutput<phi::DenseTensor>(
    const std::string& name) const;

class OpKernelBase {
 public:
  /**
   * ExecutionContext is the only parameter of Kernel Run function.
   * Run will get input/output variables, state such as momentum and
   * device resource such as CUDA stream, cublas handle, etc. from
   * ExecutionContext. User should construct it before run the Operator.
   */

  virtual void Compute(const ExecutionContext& context) const = 0;

  virtual ~OpKernelBase() = default;
};

template <typename T>
class OpKernel : public OpKernelBase {
 public:
  using ELEMENT_TYPE = T;
};

class OperatorWithKernel : public OperatorBase {
 public:
  using OpKernelFunc = std::function<void(const ExecutionContext&)>;
  using OpKernelMap =
      std::unordered_map<OpKernelType, OpKernelFunc, OpKernelType::Hash>;

  OperatorWithKernel(const std::string& type,
                     const VariableNameMap& inputs,
                     const VariableNameMap& outputs,
                     const AttributeMap& attrs);

  virtual ~OperatorWithKernel();

  static paddle::flat_hash_map<std::string /* op_type */, OpKernelMap>&
  AllOpKernels() {
    static paddle::flat_hash_map<std::string, OpKernelMap> g_all_op_kernels;
    return g_all_op_kernels;
  }

  bool SupportGPU() const override;

  bool SupportXPU() const override;

  bool SupportCustomDevice() const override;

  bool SupportsMKLDNN(phi::DataType data_type) const;

  bool SupportsCUDNN(phi::DataType data_type) const;

  bool SupportsKernelType(const OpKernelType& kernel_type,
                          const ExecutionContext& exe_ctx) const;

  bool SupportsCPUBF16() const;

  bool CanMKLDNNBeUsed(const framework::ExecutionContext& ctx,
                       phi::DataType data_type) const;

  bool CanMKLDNNBeUsed(const framework::ExecutionContext& ctx,
                       proto::VarType::Type data_type) const;

  bool CanCUDNNBeUsed(const framework::ExecutionContext& ctx,
                      phi::DataType data_type) const;

  bool CanCUDNNBeUsed(const framework::ExecutionContext& ctx,
                      proto::VarType::Type data_type) const;

  virtual void InferShape(InferShapeContext* ctx) const;

  void SetIsRuntimeInferShape(bool x) override {
    all_kernels_must_compute_runtime_shape_ = x;
  }

  void RuntimeInferShape(const Scope& scope,
                         const phi::Place& place,
                         const RuntimeContext& ctx) const override;

  proto::VarType::Type IndicateVarDataType(const ExecutionContext& ctx,
                                           const std::string& name) const;

  proto::VarType::Type IndicateOrPromoteVarDataTypes(
      const ExecutionContext& ctx,
      const std::string& name1,
      const std::string& name2) const;

  virtual phi::KernelKey GetExpectedKernelType(
      const ExecutionContext& ctx) const;

  // change this to public so that in dygraph mode we can call it to check if we
  // need transform data
  virtual phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const;

  phi::Place GetExecutionPlace(
      const phi::Place& platform UNUSED) const override {
    return kernel_type_->place_;
  }

  /* member functions for adapting to phi lib */
  /** In the phi::DenseTensor calculation library, the new Kernel adopts a
   * clearer and more streamlined design. The arguments of the Kernel and the
   * input and output arguments registered in the original OpMaker do not match
   * in some cases, so we use map to record the arguments required by the
   * kernel. When selecting Kernel during Op execution, select the arguments of
   * the original Op according to the GetExpectedPhiKernelArgs returned
   * arguments.
   */
  phi::KernelSignature GetExpectedPhiKernelArgs(
      const ExecutionContext& ctx) const;

  /* member functions for adapting to phi lib */
  phi::KernelKey ChoosePhiKernel(const ExecutionContext& ctx) const;

  void ChooseKernel(const ExecutionContext& ctx) const;

  void BuildPhiKernelContext(const RuntimeContext& ctx,
                             phi::DeviceContext* dev_ctx,
                             phi::KernelContext* phi_kernel_context) const;

  phi::KernelSignature* PhiKernelSignature() const {
    return kernel_signature_.get();
  }

  phi::Kernel* PhiKernel() const { return phi_kernel_.get(); }

  void ResetPhiKernel(phi::Kernel* kernel) const {
    return phi_kernel_.reset(kernel);
  }

  const OpKernelType* kernel_type() const { return kernel_type_.get(); }
  const OpKernelFunc* kernel_func() const { return kernel_func_.get(); }

  void ResetKernelType(OpKernelType* kernel_type) {
    kernel_type_.reset(kernel_type);
  }

  bool DnnFallback() const { return dnn_fallback_; }

  void SetDnnFallback(bool dnn_fallback) const { dnn_fallback_ = dnn_fallback; }

 private:
  void RunImpl(const Scope& scope, const phi::Place& place) const final;
  void RunImpl(const Scope& scope,
               const phi::Place& place,
               RuntimeContext* runtime_ctx) const;

  /**
   * Transfer data from scope to a transferred scope. If there is no data need
   * to be transferred, it returns nullptr.
   *
   * transferred_inplace_vars is a output vector.
   */
  Scope* PrepareData(const Scope& scope,
                     const phi::KernelKey& expected_kernel_key,
                     std::vector<std::string>* transferred_inplace_vars,
                     RuntimeContext* ctx,
                     const phi::Place& place) const;

  void CheckWhetherPreparePhiData(const VariableNameMap& innames,
                                  const VariableNameMap& outnames,
                                  const Scope& scope) const;

  void TransferInplaceVarsBack(const Scope& scope,
                               const std::vector<std::string>& inplace_vars,
                               const Scope& exec_scope) const;

  OpKernelType InnerGetExpectedKernelType(const ExecutionContext& ctx) const;

  void HandleComplexGradToRealGrad(const Scope& scope,
                                   RuntimeContext* ctx) const;

  /* Inner assist methods */
  // indicate kernel DataType by input data.
  // By default all input data must be same.
  proto::VarType::Type IndicateDataType(const ExecutionContext& ctx) const;
  // used for IndicateDataType
  void ParseInputDataType(const Variable* vars,
                          const std::string& name,
                          proto::VarType::Type* data_type) const;
  void ParseMultiInputDataType(const std::vector<Variable*>& vars,
                               const std::string& name,
                               proto::VarType::Type* data_type) const;
  // used for IndicateOrPromoteVarDataTypes
  phi::DenseTensor* GetTensorFormInputSafely(const ExecutionContext& ctx,
                                             const std::string& name) const;

 protected:
  mutable std::unique_ptr<OpKernelType> kernel_type_ = nullptr;
  mutable std::unique_ptr<OpKernelFunc> kernel_func_ = nullptr;
  mutable std::unique_ptr<RuntimeContext> runtime_ctx_ = nullptr;
  mutable const Scope* pre_scope_ = nullptr;
  mutable bool need_prepare_data_ = true;
  mutable bool need_prepare_phi_data_ = false;
  mutable bool enable_cache_runtime_context_ = false;
  mutable bool all_kernels_must_compute_runtime_shape_ = false;
  mutable std::mutex cache_update_mutex_;
  mutable bool enable_cache_transfer_scope_ = false;
  // NOTE(jiahongyu): Whether fallback to plain kernel after calling
  // GetExpectedKernelType, use this bool flag to solve onednn and cudnn hard
  // code
  mutable bool dnn_fallback_ = false;
  // NOTE(chenweihang): Similar op members are used to adapt to
  // new phi kernel, if there is a better design in the future,
  // we may polish the implementation here
  mutable bool run_phi_kernel_ = false;
  mutable bool run_kp_kernel = false;
  mutable std::unique_ptr<phi::KernelSignature> kernel_signature_ = nullptr;
  mutable std::unique_ptr<phi::Kernel> phi_kernel_ = nullptr;
  mutable std::unique_ptr<phi::ArgumentMappingFn> arg_map_fn_ = nullptr;

 private:
  struct CacheImpl;
  mutable std::unique_ptr<CacheImpl> impl_;
};

extern bool OpSupportGPU(const std::string& op_type);

}  // namespace framework
}  // namespace paddle
