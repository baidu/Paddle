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

#include "paddle/fluid/imperative/prepared_operator.h"

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/imperative/infer_shape_context.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/small_vector.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/core/platform/device/xpu/xpu_op_list.h"
#endif
#ifdef PADDLE_WITH_DNNL
#include "paddle/phi/core/platform/onednn_op_list.h"
#endif
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#elif defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_bkcl.h"
#endif
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/library_type.h"
#include "paddle/fluid/platform/profiler/supplement_tracing.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

COMMON_DECLARE_bool(check_nan_inf);
COMMON_DECLARE_bool(benchmark);
COMMON_DECLARE_bool(run_kp_kernel);

namespace paddle::imperative {

static const phi::Kernel empty_kernel;
static const framework::RuntimeContext empty_ctx({}, {});
static const framework::Scope empty_scope;

const phi::KernelFactory& PreparedOp::phi_kernel_factory =
    phi::KernelFactory::Instance();
const phi::OpUtilsMap& PreparedOp::phi_op_utils_map =
    phi::OpUtilsMap::Instance();
const phi::DefaultKernelSignatureMap& PreparedOp::default_phi_kernel_sig_map =
    phi::DefaultKernelSignatureMap::Instance();

const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<paddle::imperative::VarBase>& var) {
  return var->SharedVar();
}

const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<VariableWrapper>& var) {
  return var;
}

const phi::DenseTensor* GetTensorFromVar(const framework::Variable& var) {
  if (var.IsType<phi::DenseTensor>()) {
    return &(var.Get<phi::DenseTensor>());
  } else if (var.IsType<phi::SelectedRows>()) {
    return &(var.Get<phi::SelectedRows>().value());
  } else {
    return nullptr;
  }
}

template <typename VarType>
void HandleComplexGradToRealGrad(const NameVarMap<VarType>& outs) {
  for (auto& pair : outs) {
    for (auto& var : pair.second) {
      if (var == nullptr) {
        continue;
      }
      if (var->ForwardDataType() ==
          static_cast<framework::proto::VarType::Type>(-1)) {
        VLOG(6) << "Var (" << var->Name()
                << ")'s forward data type is not set.";
        continue;
      }
      if (!framework::IsComplexType(var->DataType()) ||
          framework::IsComplexType(var->ForwardDataType())) {
        continue;
      }
      const auto* tensor = GetTensorFromVar(var->Var());
      if (tensor && tensor->IsInitialized()) {
        VLOG(6) << "Transform " << framework::DataTypeToString(var->DataType())
                << " var `" << var->Name() << "` to "
                << framework::DataTypeToString(var->ForwardDataType())
                << " real var in dynamic graph.";
        phi::DenseTensor out;
        framework::TransComplexToReal(
            var->ForwardDataType(), var->DataType(), *tensor, &out);
        SetTensorToVariable(var->Var(), out, var->MutableVar());
      }
    }
  }
}

template <>
void HandleComplexGradToRealGrad<egr::EagerVariable>(
    const NameVarMap<egr::EagerVariable>& outs) {
  // TODO(jiabin): Support Complex here.
}

void TestHandleComplexGradToRealGradEager(
    const NameVarMap<egr::EagerVariable>& outs) {
  HandleComplexGradToRealGrad<egr::EagerVariable>(outs);
}

PreparedOp::PreparedOp(const framework::OperatorBase& op,
                       const framework::RuntimeContext& ctx,
                       const phi::KernelKey& kernel_key,
                       const framework::OperatorWithKernel::OpKernelFunc& func,
                       const phi::ArgumentMappingFn* arg_map_fn,
                       const phi::KernelSignature* default_kernel_signature,
                       phi::DeviceContext* dev_ctx)
    : op_(op),
      ctx_(ctx),
      kernel_key_(kernel_key),
      func_(func),
      dev_ctx_(dev_ctx),
      arg_map_fn_(arg_map_fn),
      default_kernel_signature_(default_kernel_signature),
      phi_kernel_(empty_kernel) {}

PreparedOp::PreparedOp(const framework::OperatorBase& op,
                       const framework::RuntimeContext& ctx,
                       const phi::KernelKey& kernel_key,
                       const phi::ArgumentMappingFn* arg_map_fn,
                       const phi::KernelSignature* default_kernel_signature,
                       phi::KernelSignature&& kernel_signature,
                       const phi::Kernel& phi_kernel,
                       phi::DeviceContext* dev_ctx)
    : op_(op),
      ctx_(ctx),
      kernel_key_(kernel_key),
      func_(nullptr),
      dev_ctx_(dev_ctx),
      run_phi_kernel_(true),
      arg_map_fn_(arg_map_fn),
      default_kernel_signature_(default_kernel_signature),
      kernel_signature_(std::move(kernel_signature)),
      phi_kernel_(phi_kernel) {}

template <typename VarType>
PreparedOp PrepareImpl(
    const NameVarMap<VarType>& ins,
    const NameVarMap<VarType>& outs,
    const framework::OperatorWithKernel& op,
    const phi::Place& place,
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs,
    const phi::KernelFactory& phi_kernel_factory,
    const phi::OpUtilsMap& phi_op_utils_map,
    const phi::DefaultKernelSignatureMap& default_phi_kernel_sig_map) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

#ifdef PADDLE_WITH_DNNL
  // OneDNN variant of code reads attributes in some of GetKernelTypeForVar and
  // GetKernelType functions, so we need to copy the attributes there.
  // Const qualifier of Attrs had to be discarded to overwrite it.
  if (FLAGS_use_mkldnn) {
    auto& mutable_op_attrs = const_cast<framework::AttributeMap&>(op.Attrs());
    mutable_op_attrs = default_attrs;
    for (auto& attr : attrs) {
      mutable_op_attrs[attr.first] = attr.second;
    }
  }
#endif
  // NOTE(zhiqiu): for kernels on given device, for example NPU, the order to
  // choose is:
  // phi npu kernel > fluid npu kernel > phi cpu kernel > fluid cpu kernel

  // 1. get expected kernel key
  auto dygraph_exe_ctx = DygraphExecutionContext<VarType>(
      op, empty_scope, *dev_ctx, empty_ctx, ins, outs, attrs, default_attrs);
  auto expected_kernel_key = op.GetExpectedKernelType(dygraph_exe_ctx);

  const phi::KernelSignature* default_kernel_signature = nullptr;
  phi::KernelSignature kernel_signature;
  std::string phi_kernel_name;

// NOTE(jiahongyu): The registered OneDNN kernel have library_type =
// LibraryType::kMKLDNN and data_layout_ = DataLayout::ONEDNN. But the default
// values are kPlain, so we need to modify the library_type and data_layout_
// here. There are three statements in if condition:
// 1. Whether onednn kernel fallbacks to plain kernel;
// 2. Whether this op has specific implementation;
// 3. Whether onednn kernel can be used.
#ifdef PADDLE_WITH_DNNL
  if (!op.DnnFallback() && !paddle::platform::in_onednn_white_list(op.Type()) &&
      op.CanMKLDNNBeUsed(dygraph_exe_ctx, expected_kernel_key.dtype())) {
    expected_kernel_key.set_backend(phi::Backend::ONEDNN);
    expected_kernel_key.set_layout(phi::DataLayout::ONEDNN);
  }
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (op.CanCUDNNBeUsed(dygraph_exe_ctx, expected_kernel_key.dtype())) {
    expected_kernel_key.set_backend(phi::Backend::GPUDNN);
  }
#endif

#if defined(PADDLE_WITH_XPU)
  bool is_xpu_unsupported =
      expected_kernel_key.backend() == phi::Backend::XPU &&
      !paddle::platform::is_xpu_support_op(op.Type(),
                                           expected_kernel_key.dtype());
#endif

  bool has_phi_kernel = false;

  const auto* arg_map_fn = phi_op_utils_map.GetArgumentMappingFn(op.Type());

  if (arg_map_fn) {
    has_phi_kernel = true;
    kernel_signature = (*arg_map_fn)(
        framework::ExecutionArgumentMappingContext(dygraph_exe_ctx));
  } else {
    if (phi::KernelFactory::Instance().HasStructuredKernel(op.Type())) {
      has_phi_kernel = true;
      kernel_signature = phi::KernelSignature(op.Type().c_str());
    } else {
      default_kernel_signature =
          default_phi_kernel_sig_map.GetNullable(op.Type());
      if (default_kernel_signature) {
        has_phi_kernel = true;
        kernel_signature = *default_kernel_signature;
      }
    }
  }

  if (has_phi_kernel) {
    VLOG(6) << kernel_signature;
    phi_kernel_name = kernel_signature.name;

// NOTE(Liu-xiandong): The register kernel used KP have library_type[KP],
// But the default library_type is Plain, so we need to modify the
// library_type here, otherwise it can't work.
#ifdef PADDLE_WITH_XPU_KP
    if (expected_kernel_key.backend() == phi::Backend::XPU) {
      bool use_xpu_kp_kernel_rt =
          FLAGS_run_kp_kernel && paddle::platform::is_xpu_kp_support_op(
                                     op.Type(), expected_kernel_key.dtype());
      bool use_xpu_kp_kernel_debug =
          paddle::platform::is_in_xpu_kpwhite_list(op.Type());
      if (use_xpu_kp_kernel_rt) {
        VLOG(3) << "phi xpu_kp using rt mode ";
      }
      if (use_xpu_kp_kernel_debug) {
        VLOG(3) << "phi xpu_kp using debug mode ";
      }
      bool is_xpu_kp_support =
          (use_xpu_kp_kernel_rt || use_xpu_kp_kernel_debug);
      if (is_xpu_kp_support) {
        auto expected_kernel_key_backend = expected_kernel_key.backend();
        expected_kernel_key.set_backend(phi::Backend::KPS);
        VLOG(3) << "modifying XPU KP kernel: " << phi_kernel_name
                << ", using_kernel_key:" << expected_kernel_key;

        if (!phi_kernel_factory.HasKernel(phi_kernel_name,
                                          expected_kernel_key)) {
          expected_kernel_key.set_backend(expected_kernel_key_backend);
          VLOG(3) << "modify XPU KP kernel: " << phi_kernel_name
                  << " in dynamic graph is failed " << expected_kernel_key;
        } else {
          VLOG(3) << "modify XPU KP kernel: " << phi_kernel_name
                  << " in dynamic graph is succeed " << expected_kernel_key;
        }
      }
    }
#endif

    auto& phi_kernel =
        phi_kernel_factory.SelectKernel(phi_kernel_name, expected_kernel_key);

    if (phi_kernel.IsValid()
#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
        && !is_xpu_unsupported
#endif
    ) {
      VLOG(6) << "Dynamic mode PrepareImpl - kernel name: " << phi_kernel_name
              << " | kernel key: " << expected_kernel_key
              << " | kernel: " << phi_kernel;

      if (!framework::backends_are_same_class(
              expected_kernel_key.backend(),
              phi::TransToPhiBackend(dev_ctx->GetPlace()))) {
        dev_ctx = pool.Get(phi::TransToPhiPlace(expected_kernel_key.backend()));
      }
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      if (attrs.find("ring_id") != attrs.end()) {
        auto ring_id_attr = attrs.at("ring_id");
        int ring_id = PADDLE_GET(int, ring_id_attr);
        const auto& comm_context_manager =
            phi::distributed::CommContextManager::GetInstance();
        auto map = distributed::ProcessGroupMapFromGid::getInstance();
        phi::distributed::CommContext* comm_context = nullptr;
        if (comm_context_manager.Has(std::to_string(ring_id))) {
          comm_context = comm_context_manager.Get(std::to_string(ring_id));
        } else if (map->has(ring_id)) {
          distributed::ProcessGroup* pg = map->get(ring_id);
          comm_context = static_cast<paddle::distributed::ProcessGroupNCCL*>(pg)
                             ->GetOrCreateCommContext(place);
        }
        if (comm_context) {
          auto original_stream =
              static_cast<phi::GPUContext*>(dev_ctx)->cuda_stream();
          dev_ctx =
              static_cast<phi::distributed::NCCLCommContext*>(comm_context)
                  ->GetDevContext();
          dev_ctx->SetCommContext(comm_context);
          // Note(lizhenxing): In dynamic mode, c_softmax_with_cross_entropy
          // need use global calculate stream (original_stream). Using the
          // comm_ctx's stream will lead to synchronization issues, causing
          // accuracy diff in test_parallel_dygraph_mp_layers.
          if (phi::is_gpu_place(place) &&
              ((attrs.find("use_calc_stream") != attrs.end() &&
                PADDLE_GET_CONST(bool, attrs.at("use_calc_stream"))) ||
               phi_kernel_name == "c_softmax_with_cross_entropy" ||
               phi_kernel_name == "c_softmax_with_multi_label_cross_entropy")) {
            static_cast<phi::GPUContext*>(dev_ctx)->SetCUDAStream(
                original_stream, false);
            auto& instance =
                paddle::memory::allocation::AllocatorFacade::Instance();
            dev_ctx->SetAllocator(
                instance
                    .GetAllocator(
                        place, static_cast<phi::GPUContext*>(dev_ctx)->stream())
                    .get());
          }
        }
      }
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
      if (attrs.find("ring_id") != attrs.end()) {
        auto ring_id_attr = attrs.at("ring_id");
        int ring_id = PADDLE_GET(int, ring_id_attr);
        auto map = distributed::ProcessGroupMapFromGid::getInstance();
        if (map->has(ring_id)) {
          distributed::ProcessGroup* pg = map->get(ring_id);
          auto comm_context =
              static_cast<paddle::distributed::ProcessGroupBKCL*>(pg)
                  ->GetOrCreateCommContext(place);
          auto original_stream =
              static_cast<phi::XPUContext*>(dev_ctx)->stream();
          dev_ctx =
              static_cast<phi::distributed::BKCLCommContext*>(comm_context)
                  ->GetDevContext();
          dev_ctx->SetCommContext(comm_context);
          // Note(lizhenxing): In dynamic mode, c_softmax_with_cross_entropy
          // need use global calculate stream (original_stream). Using the
          // comm_ctx's stream will lead to synchronization issues, causing
          // accuracy diff in test_parallel_dygraph_mp_layers.
          if (phi::is_xpu_place(place) &&
              ((attrs.find("use_calc_stream") != attrs.end() &&
                PADDLE_GET_CONST(bool, attrs.at("use_calc_stream"))) ||
               phi_kernel_name == "c_softmax_with_cross_entropy" ||
               phi_kernel_name == "c_softmax_with_multi_label_cross_entropy")) {
            static_cast<phi::XPUContext*>(dev_ctx)->SetStream(original_stream,
                                                              false);
            auto& instance =
                paddle::memory::allocation::AllocatorFacade::Instance();
            dev_ctx->SetAllocator(
                instance
                    .GetAllocator(
                        place, static_cast<phi::XPUContext*>(dev_ctx)->stream())
                    .get());
          }
        }
      }
#endif
      return PreparedOp(op,
                        empty_ctx,
                        expected_kernel_key,
                        arg_map_fn,
                        default_kernel_signature,
                        std::move(kernel_signature),
                        phi_kernel,
                        dev_ctx);
    } else {
      VLOG(6) << "Dynamic mode ChoosePhiKernel - kernel `" << phi_kernel_name
              << "` not found.";
    }
  }

  // 2. check if op[type] has kernel registered.
  auto& all_op_kernels = op.AllOpKernels();
  auto kernels_iter = all_op_kernels.find(op.Type());

// NOTE(Liu-xiandong): If we can't find heterogeneous kernel in phi,
// we need to select the heterogeneous kernel in fluid, but the kernel
// registered in KP use library_type[KP], we need to modify it.
#ifdef PADDLE_WITH_XPU_KP
  bool use_xpu_kp_kernel_rt =
      expected_kernel_key.backend() == phi::Backend::XPU &&
      FLAGS_run_kp_kernel &&
      paddle::platform::is_xpu_kp_support_op(op.Type(),
                                             expected_kernel_key.dtype());
  bool use_xpu_kp_kernel_debug =
      expected_kernel_key.backend() == phi::Backend::XPU &&
      paddle::platform::is_in_xpu_kpwhite_list(op.Type());
  bool is_xpu_kp_support = (use_xpu_kp_kernel_rt || use_xpu_kp_kernel_debug);
  if (is_xpu_kp_support) {
    expected_kernel_key.set_backend(phi::Backend::KPS);
  }
#endif

  paddle::framework::OpKernelType fluid_kernel_type =
      paddle::framework::TransPhiKernelKeyToOpKernelType(expected_kernel_key);
  if ((kernels_iter == all_op_kernels.end() ||
       kernels_iter->second.find(fluid_kernel_type) ==
           kernels_iter->second.end())
#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
      || is_xpu_unsupported
#endif
#if defined(PADDLE_WITH_XPU_KP)
      || (is_xpu_unsupported && !is_xpu_kp_support)
#endif
  ) {
    if (has_phi_kernel) {
      auto phi_cpu_kernel_key = FallBackToCpu(expected_kernel_key, op);
      auto& phi_cpu_kernel =
          phi_kernel_factory.SelectKernel(phi_kernel_name, phi_cpu_kernel_key);
      if (phi_cpu_kernel.IsValid()) {
        VLOG(6) << "Dynamic mode PrepareImpl - kernel name: " << phi_kernel_name
                << " | kernel key: " << phi_cpu_kernel_key
                << " | kernel: " << phi_cpu_kernel;
        auto* cpu_ctx = pool.Get(phi::CPUPlace());
        return PreparedOp(op,
                          empty_ctx,
                          phi_cpu_kernel_key,
                          arg_map_fn,
                          default_kernel_signature,
                          std::move(kernel_signature),
                          phi_cpu_kernel,
                          cpu_ctx);
      }
    }
  }

  PADDLE_ENFORCE_NE(
      kernels_iter,
      all_op_kernels.end(),
      common::errors::NotFound(
          "There are no kernels which are registered in the %s operator.",
          op.Type()));

  auto& kernels = kernels_iter->second;
  auto kernel_iter = kernels.find(fluid_kernel_type);

#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
  if (phi::is_xpu_place(fluid_kernel_type.place_) &&
      (kernel_iter == kernels.end() || is_xpu_unsupported)) {
    VLOG(3) << "fluid missing XPU kernel: " << op.Type()
            << ", expected_kernel_key:" << fluid_kernel_type
            << ", fallbacking to CPU one!";
    fluid_kernel_type.place_ = phi::CPUPlace();
    kernel_iter = kernels.find(fluid_kernel_type);
  }
#endif

#ifdef PADDLE_WITH_XPU_KP
  if (phi::is_xpu_place(fluid_kernel_type.place_)) {
    if (use_xpu_kp_kernel_rt) {
      VLOG(3) << "fluid xpu_kp using rt mode ";
    }
    if (use_xpu_kp_kernel_debug) {
      VLOG(3) << "fluid xpu_kp using debug mode ";
    }
    if (is_xpu_kp_support) {
      fluid_kernel_type.library_type_ = paddle::framework::LibraryType::kKP;
      kernel_iter = kernels.find(fluid_kernel_type);
      VLOG(3) << "using fluid XPU KP kernel: " << op.Type()
              << ", using_kernel_key:" << fluid_kernel_type;
    }
    if (!is_xpu_kp_support &&
        (kernel_iter == kernels.end() || is_xpu_unsupported)) {
      VLOG(3) << "fluid missing XPU kernel: " << op.Type()
              << ", expected_kernel_key:" << fluid_kernel_type
              << ", fallbacking to CPU one!";
      fluid_kernel_type.place_ = phi::CPUPlace();
      kernel_iter = kernels.find(fluid_kernel_type);
    }
  }
#endif
#ifdef PADDLE_WITH_IPU
  if (kernel_iter == kernels.end() &&
      phi::is_ipu_place(fluid_kernel_type.place_)) {
    VLOG(3) << "missing IPU kernel: " << op.Type()
            << ", expected_kernel_key:" << fluid_kernel_type
            << ", fallbacking to CPU one!";
    fluid_kernel_type.place_ = phi::CPUPlace();
    kernel_iter = kernels.find(fluid_kernel_type);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (kernel_iter == kernels.end() &&
      phi::is_custom_place(fluid_kernel_type.place_)) {
    VLOG(3) << "missing " << place.GetDeviceType() << " kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    fluid_kernel_type.place_ = phi::CPUPlace();
    kernel_iter = kernels.find(fluid_kernel_type);
  }
#endif
  // TODO(jiabin): Add operator.cc's line 1000 part back when we need that
  // case
  PADDLE_ENFORCE_NE(
      kernel_iter,
      kernels.end(),
      common::errors::NotFound("Operator %s does not have kernel for %s.",
                               op.Type(),
                               KernelTypeToString(fluid_kernel_type)));

  if (!phi::places_are_same_class(fluid_kernel_type.place_,
                                  dev_ctx->GetPlace())) {
    dev_ctx = pool.Get(fluid_kernel_type.place_);
  }
  return PreparedOp(
      op,
      empty_ctx,
      framework::TransOpKernelTypeToPhiKernelKey(fluid_kernel_type),
      kernel_iter->second,
      arg_map_fn,
      default_kernel_signature,
      dev_ctx);
}

PreparedOp PreparedOp::Prepare(const NameVarMap<VarBase>& ins,
                               const NameVarMap<VarBase>& outs,
                               const framework::OperatorWithKernel& op,
                               const phi::Place& place,
                               const framework::AttributeMap& attrs,
                               const framework::AttributeMap& default_attrs) {
  return PrepareImpl<VarBase>(ins,
                              outs,
                              op,
                              place,
                              attrs,
                              default_attrs,
                              phi_kernel_factory,
                              phi_op_utils_map,
                              default_phi_kernel_sig_map);
}

PreparedOp PreparedOp::Prepare(const NameVarMap<VariableWrapper>& ins,
                               const NameVarMap<VariableWrapper>& outs,
                               const framework::OperatorWithKernel& op,
                               const phi::Place& place,
                               const framework::AttributeMap& attrs,
                               const framework::AttributeMap& default_attrs) {
  return PrepareImpl<VariableWrapper>(ins,
                                      outs,
                                      op,
                                      place,
                                      attrs,
                                      default_attrs,
                                      phi_kernel_factory,
                                      phi_op_utils_map,
                                      default_phi_kernel_sig_map);
}

PreparedOp PreparedOp::Prepare(const NameVarMap<egr::EagerVariable>& ins,
                               const NameVarMap<egr::EagerVariable>& outs,
                               const framework::OperatorWithKernel& op,
                               const phi::Place& place,
                               const framework::AttributeMap& attrs,
                               const framework::AttributeMap& default_attrs) {
  return PrepareImpl<egr::EagerVariable>(ins,
                                         outs,
                                         op,
                                         place,
                                         attrs,
                                         default_attrs,
                                         phi_kernel_factory,
                                         phi_op_utils_map,
                                         default_phi_kernel_sig_map);
}
template <typename VarType>
static void PreparedOpRunImpl(
    const framework::OperatorBase& op,
    const framework::RuntimeContext& ctx,
    const phi::KernelKey& kernel_key,
    const framework::OperatorWithKernel::OpKernelFunc& func,
    const phi::ArgumentMappingFn* arg_map_fn,
    const phi::KernelSignature* default_kernel_signature,
    phi::DeviceContext* dev_ctx,
    const NameVarMap<VarType>& ins,
    const NameVarMap<VarType>& outs,
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs) {
  // TODO(zjl): remove scope in dygraph

  {
    phi::RecordEvent record_event("infer_shape",
                                  phi::TracerEventType::OperatorInner,
                                  1,
                                  phi::EventRole::kInnerOp);
    DygraphInferShapeContext<VarType> infer_shape_ctx(&ins,
                                                      &outs,
                                                      &attrs,
                                                      &default_attrs,
                                                      op.Type(),
                                                      &kernel_key,
                                                      arg_map_fn,
                                                      default_kernel_signature);
    op.Info().infer_shape_(&infer_shape_ctx);
    record_event.End();
    platform::RecordOpInfoSupplement(
        op.Type(), op.Attrs(), infer_shape_ctx, ctx, op.Id());
  }

  {
    phi::RecordEvent record_event("compute",
                                  phi::TracerEventType::OperatorInner,
                                  1,
                                  phi::EventRole::kInnerOp);

    func(DygraphExecutionContext<VarType>(
        op, empty_scope, *dev_ctx, ctx, ins, outs, attrs, default_attrs));
  }

  if (FLAGS_check_nan_inf) {
    framework::details::CheckOpHasNanOrInfInDygraph<VarType>(
        op.Type(), outs, dev_ctx->GetPlace());
  }

  if (FLAGS_benchmark) {
    dev_ctx->Wait();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_GPU_SUCCESS(platform::GpuGetLastError());
    VLOG(4) << "Operator(" << op.Type() << "): context wait and get last error";
#endif
  }

  /**
   * [ Why need handle complex gradient to real gradient? ]
   *
   * After the introduction of complex number calculations, Ops that support
   * complex number calculations generally support type promotion, such as
   * x(float32) + y(complex64) = out(complex64), then the type of the grad
   * tensor should be dout(complex64), dx(float32), dy (complex64).
   *
   * But because the dout is complex64, the dx is also complex64 after
   * grad op kernel executed, we need to recognize this situation and
   * convert dx to float32 type. HandleComplexGradToRealGrad does this thing.
   */
  if (framework::IsComplexType(kernel_key.dtype())) {
    HandleComplexGradToRealGrad<VarType>(outs);
  }
}

template <typename VarType>
static void PreparedOpRunPtImpl(
    const framework::OperatorBase& op,
    const phi::KernelKey& kernel_key,
    const phi::ArgumentMappingFn* arg_map_fn,
    const phi::KernelSignature* default_kernel_signature,
    const phi::KernelSignature& kernel_signature,
    const phi::Kernel& phi_kernel,
    const framework::RuntimeContext& ctx,
    phi::DeviceContext* dev_ctx,
    const NameVarMap<VarType>& ins,
    const NameVarMap<VarType>& outs,
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs) {
  {
    phi::RecordEvent record_event("infer_shape",
                                  phi::TracerEventType::OperatorInner,
                                  1,
                                  phi::EventRole::kInnerOp);
    DygraphInferShapeContext<VarType> infer_shape_ctx(&ins,
                                                      &outs,
                                                      &attrs,
                                                      &default_attrs,
                                                      op.Type(),
                                                      &kernel_key,
                                                      arg_map_fn,
                                                      default_kernel_signature);
    op.Info().infer_shape_(&infer_shape_ctx);
    record_event.End();
    platform::RecordOpInfoSupplement(
        op.Type(), op.Attrs(), infer_shape_ctx, kernel_signature);
  }

  {
    phi::RecordEvent record_event("compute",
                                  phi::TracerEventType::OperatorInner,
                                  1,
                                  phi::EventRole::kInnerOp);

    if (phi_kernel.GetKernelRegisteredType() ==
        phi::KernelRegisteredType::FUNCTION) {
      PreparePhiData<VarType>(phi_kernel, kernel_signature, ins);
      phi::KernelContext phi_kernel_context;
      BuildDygraphPhiKernelContext<VarType>(kernel_signature,
                                            phi_kernel,
                                            ins,
                                            outs,
                                            attrs,
                                            default_attrs,
                                            dev_ctx,
                                            &phi_kernel_context);

      phi_kernel(&phi_kernel_context);
    } else {
      DygraphExecutionContext<VarType> exe_ctx(
          op, empty_scope, *dev_ctx, ctx, ins, outs, attrs, default_attrs);
      phi_kernel(&exe_ctx);
    }
  }

  if (FLAGS_check_nan_inf) {
    framework::details::CheckOpHasNanOrInfInDygraph<VarType>(
        op.Type(), outs, dev_ctx->GetPlace());
  }

  if (FLAGS_benchmark) {
    dev_ctx->Wait();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_GPU_SUCCESS(platform::GpuGetLastError());
    VLOG(4) << "Operator(" << op.Type() << "): context wait and get last error";
#endif
  }

  if (framework::IsComplexType(kernel_key.dtype())) {
    HandleComplexGradToRealGrad<VarType>(outs);
  }
}

void PreparedOp::Run(const NameVarMap<VarBase>& ins,
                     const NameVarMap<VarBase>& outs,
                     const framework::AttributeMap& attrs,
                     const framework::AttributeMap& default_attrs) {
  if (run_phi_kernel_) {  // NOLINT
    PreparedOpRunPtImpl<VarBase>(op_,
                                 kernel_key_,
                                 arg_map_fn_,
                                 default_kernel_signature_,
                                 kernel_signature_,
                                 phi_kernel_,
                                 ctx_,
                                 dev_ctx_,
                                 ins,
                                 outs,
                                 attrs,
                                 default_attrs);
  } else {
    PreparedOpRunImpl<VarBase>(op_,
                               ctx_,
                               kernel_key_,
                               func_,
                               arg_map_fn_,
                               default_kernel_signature_,
                               dev_ctx_,
                               ins,
                               outs,
                               attrs,
                               default_attrs);
  }
}

void PreparedOp::Run(const NameVarMap<VariableWrapper>& ins,
                     const NameVarMap<VariableWrapper>& outs,
                     const framework::AttributeMap& attrs,
                     const framework::AttributeMap& default_attrs) {
  if (run_phi_kernel_) {  // NOLINT
    PreparedOpRunPtImpl<VariableWrapper>(op_,
                                         kernel_key_,
                                         arg_map_fn_,
                                         default_kernel_signature_,
                                         kernel_signature_,
                                         phi_kernel_,
                                         ctx_,
                                         dev_ctx_,
                                         ins,
                                         outs,
                                         attrs,
                                         default_attrs);
  } else {
    PreparedOpRunImpl<VariableWrapper>(op_,
                                       ctx_,
                                       kernel_key_,
                                       func_,
                                       arg_map_fn_,
                                       default_kernel_signature_,
                                       dev_ctx_,
                                       ins,
                                       outs,
                                       attrs,
                                       default_attrs);
  }
}

void PreparedOp::Run(const NameVarMap<egr::EagerVariable>& ins,
                     const NameVarMap<egr::EagerVariable>& outs,
                     const framework::AttributeMap& attrs,
                     const framework::AttributeMap& default_attrs) {
  if (run_phi_kernel_) {  // NOLINT
    PreparedOpRunPtImpl<egr::EagerVariable>(op_,
                                            kernel_key_,
                                            arg_map_fn_,
                                            default_kernel_signature_,
                                            kernel_signature_,
                                            phi_kernel_,
                                            ctx_,
                                            dev_ctx_,
                                            ins,
                                            outs,
                                            attrs,
                                            default_attrs);
  } else {
    PreparedOpRunImpl<egr::EagerVariable>(op_,
                                          ctx_,
                                          kernel_key_,
                                          func_,
                                          arg_map_fn_,
                                          default_kernel_signature_,
                                          dev_ctx_,
                                          ins,
                                          outs,
                                          attrs,
                                          default_attrs);
  }
}

}  // namespace paddle::imperative
