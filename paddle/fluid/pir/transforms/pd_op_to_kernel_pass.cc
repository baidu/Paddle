// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"

#include <iostream>
#include <regex>
#include <string>
#include <unordered_set>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/new_executor/collect_shape_manager.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/interface/parse_kernel_key.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/tensorrt_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/backends/custom/custom_device_op_list.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/fluid/custom_engine/custom_engine_manager.h"
#endif

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_onednn_dialect.h"
#include "paddle/fluid/pir/dialect/operator/trait/onednn.h"
#include "paddle/phi/core/framework/framework.pb.h"
COMMON_DECLARE_bool(use_mkldnn);
#endif

COMMON_DECLARE_bool(print_ir);
COMMON_DECLARE_bool(enable_collect_shape);
REGISTER_FILE_SYMBOLS(pd_op_to_kernel_pass);
namespace paddle::dialect {

pir::Type ConvertOpTypeToKernelType(pir::IrContext* ctx,
                                    pir::Type op_type,
                                    phi::Place place) {
  if (op_type.isa<DenseTensorType>()) {
    return AllocatedDenseTensorType::get(
        ctx, place, op_type.dyn_cast<DenseTensorType>());
  } else if (op_type.isa<DenseTensorArrayType>()) {
    return AllocatedDenseTensorArrayType::get(
        ctx, place, op_type.dyn_cast<DenseTensorArrayType>());
  } else if (op_type.isa<SparseCooTensorType>()) {
    return AllocatedSparseCooTensorType::get(
        ctx, place, op_type.dyn_cast<SparseCooTensorType>());
  } else if (op_type.isa<SparseCsrTensorType>()) {
    return AllocatedSparseCsrTensorType::get(
        ctx, place, op_type.dyn_cast<SparseCsrTensorType>());
  } else if (op_type.isa<SelectedRowsType>()) {
    return AllocatedSelectedRowsType::get(
        ctx, place, op_type.dyn_cast<SelectedRowsType>());
  } else if (op_type.isa<pir::VectorType>()) {
    auto vec_type = op_type.dyn_cast<pir::VectorType>();
    std::vector<pir::Type> vec_target_type;
    for (size_t i = 0; i < vec_type.size(); ++i) {
      vec_target_type.push_back(
          ConvertOpTypeToKernelType(ctx, vec_type[i], place));
    }
    return pir::VectorType::get(ctx, vec_target_type);
  }
  PADDLE_THROW(common::errors::Unimplemented(
      "Not support op type %s in ConvertOpTypeToKernelType.", op_type));
}

static const std::vector<pir::Type> InferMetaByValue(
    pir::Operation* op,
    const std::vector<pir::Value>& input_values,
    pir::AttributeMap* p_attribute_map) {  // NOLINT
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op->name());
  auto infer_meta_interface =
      op_info.GetInterfaceImpl<paddle::dialect::InferMetaInterface>();
  std::vector<pir::Type> output_types;
  if (infer_meta_interface) {
    output_types = infer_meta_interface->infer_meta_by_value_(input_values,
                                                              p_attribute_map);
  }
  return output_types;
}

std::unordered_map<std::string, phi::DataType> Str2PhiDataType = {
    {"DataType::FLOAT16", phi::DataType::FLOAT16},
    {"DataType::BFLOAT16", phi::DataType::BFLOAT16},
    {"DataType::FLOAT32", phi::DataType::FLOAT32},
    {"DataType::FLOAT64", phi::DataType::FLOAT64},
    {"DataType::INT16", phi::DataType::INT16},
    {"DataType::INT32", phi::DataType::INT32},
    {"DataType::INT64", phi::DataType::INT64},
    {"DataType::INT8", phi::DataType::INT8},
    {"DataType::BOOL", phi::DataType::BOOL},
};

const std::unordered_set<std::string> UnchangeOutputOps = {
    pir::CombineOp::name(),
    pir::SliceOp::name(),
    pir::SplitOp::name(),
    pir::ConstantTensorOp::name(),
    pir::SetParameterOp::name(),
    pir::ParameterOp::name(),
    pir::ShadowOutputOp::name(),
    FeedOp::name(),
    DataOp::name(),
    ArrayLengthOp::name(),
    "cinn_runtime.jit_kernel"};
const std::unordered_set<std::string> SpecialLowerOps = {
    pir::CombineOp::name(),
    pir::ConstantTensorOp::name(),
    pir::SetParameterOp::name(),
    pir::ParameterOp::name(),
    pir::ShadowOutputOp::name(),
    pir::SliceOp::name(),
    pir::SplitOp::name(),
    pir::YieldOp::name(),
    IfOp::name(),
    WhileOp::name(),
    PyLayerOp::name(),
    pir::StackCreateOp::name(),
    pir::TuplePushOp::name(),
    pir::TuplePopOp::name(),
    HasElementsOp::name(),
    AssertOp::name(),
    SelectInputOp::name(),
    SelectOutputOp::name(),
    "cinn_runtime.jit_kernel"};

const std::unordered_map<std::string, uint32_t> NoBufferRelatedOps = {
    {paddle::dialect::ReshapeOp::name(), /*xshape_idx*/ 1U},
    {paddle::dialect::Reshape_Op::name(), /*xshape_idx*/ 1U},
    {paddle::dialect::SqueezeOp::name(), /*xshape_idx*/ 1U},
    {paddle::dialect::Squeeze_Op::name(), /*xshape_idx*/ 1U},
    {paddle::dialect::UnsqueezeOp::name(), /*xshape_idx*/ 1U},
    {paddle::dialect::Unsqueeze_Op::name(), /*xshape_idx*/ 1U},
    {paddle::dialect::FlattenOp::name(), /*xshape_idx*/ 1U},
    {paddle::dialect::Flatten_Op::name(), /*xshape_idx*/ 1U},
    {paddle::dialect::BatchNormOp::name(), /*reserve_space*/ 5U},
    {paddle::dialect::BatchNorm_Op::name(), /*reserve_space*/ 5U},
};

// Please keep the consistency with paddle/phi/kernels/memcpy_kernel.cc
const std::unordered_map<int, phi::Place> MemcpyOpAttr2Place = {
    {0, phi::CPUPlace()},
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    {1, phi::GPUPlace()},
    {2, phi::GPUPinnedPlace()},
#elif defined(PADDLE_WITH_XPU)
    {3, phi::XPUPlace()},
#elif defined(PADDLE_WITH_CUSTOM_DEVICE)
    {4, phi::CustomPlace()}
#endif
};

static bool NeedSkipPlaceTransfer(const pir::Operation* op) {
  bool need_skip = false;
  if (op->isa<paddle::dialect::FetchOp>()) {
    auto define_op_name = op->operand_source(0).defining_op()->name();
    uint32_t index = op->operand_source(0).dyn_cast<pir::OpResult>().index();
    need_skip = NoBufferRelatedOps.count(define_op_name) > 0 &&
                (NoBufferRelatedOps.at(define_op_name) == index);
  }
  return need_skip;
}

static bool NeedFallBackCpu(const pir::Operation* op,
                            const std::string& kernel,
                            const phi::KernelKey& kernel_key) {
  if (op->HasAttribute(kForceBackendAttr) &&
      op->attributes()
              .at(kForceBackendAttr)
              .dyn_cast<pir::StrAttribute>()
              .AsString() == "cpu") {
    return true;
  }

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  if (phi::backends::custom_device::is_in_custom_black_list(
          phi::TransToFluidOpName(kernel))) {
    phi::KernelKey copy_key = kernel_key;
    copy_key.set_backend(phi::Backend::CPU);
    if (phi::KernelFactory::Instance().HasKernel(kernel, copy_key)) {
      return true;
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Can not fallback %s from custom_device to CPU, no CPU kernel found "
          "for kernel key = %s.",
          kernel,
          copy_key));
    }
  }
#endif

  if (UnchangeOutputOps.count(op->name()) || kernel == "" ||
      phi::KernelFactory::Instance().HasKernel(kernel, kernel_key)) {
    return false;
  }

  phi::KernelKey copy_key = kernel_key;
  if (copy_key.backend() == phi::Backend::GPUDNN) {
    copy_key.set_backend(phi::Backend::GPU);
    if (phi::KernelFactory::Instance().HasKernel(kernel, copy_key)) {
      return false;
    }
  }
  copy_key.set_backend(phi::Backend::CPU);
  if (phi::KernelFactory::Instance().HasKernel(kernel, copy_key)) {
    return true;
  }

  return false;
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
static bool NeedFallBackFromGPUDNN2GPU(pir::Operation* op,
                                       const std::string& kernel_name,
                                       const phi::KernelKey kernel_key) {
  if (op->HasAttribute(kForceBackendAttr) &&
      op->attributes()
              .at(kForceBackendAttr)
              .dyn_cast<pir::StrAttribute>()
              .AsString() == "gpu") {
    return true;
  }

  // NOTE(phlrain): keep the same kernel select strategy with
  // GetExpectKernelKey
  if (op->isa<Pool2dOp>() || op->isa<Pool2dGradOp>() || op->isa<Pool3dOp>() ||
      op->isa<Pool3dGradOp>()) {
    if (kernel_key.backend() == phi::Backend::GPUDNN &&
        (op->attributes()
             .at("adaptive")
             .dyn_cast<pir::BoolAttribute>()
             .data() == true)) {
      return true;
    }
  } else if ((op->isa<AffineGridOp>() || op->isa<AffineGridGradOp>()) &&
             kernel_key.backend() == phi::Backend::GPUDNN) {
    bool use_cudnn = true;
    int version = -1;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    version = platform::DnnVersion();
#endif
    if (version >= 6000 && op->attributes()
                                   .at("align_corners")
                                   .dyn_cast<pir::BoolAttribute>()
                                   .data() == true) {
      use_cudnn = true;
    } else {
      use_cudnn = false;
    }

    auto shape = pir::GetShapeFromValue(op->operand_source(0));
    if (shape[1] == 3) {
      use_cudnn = false;
    }
#if defined(PADDLE_WITH_HIP)
    use_cudnn = false;
#endif
    return !use_cudnn;
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (kernel_key.backend() == phi::Backend::GPUDNN) {
    auto iter = phi::KernelFactory::Instance().kernels().find(kernel_name);
    if (iter != phi::KernelFactory::Instance().kernels().end()) {
      auto kernel_iter = iter->second.find({phi::Backend::GPUDNN,
                                            phi::DataLayout::ALL_LAYOUT,
                                            kernel_key.dtype()});
      if (kernel_iter == iter->second.end()) {
        return true;
      }
    }
  }
#endif

  return false;
}
#endif

static phi::Backend DeriveBackend(const std::string& op,
                                  const phi::Place& place,
                                  const OpYamlInfoParser* op_info_parser,
                                  phi::Backend kernel_backend,
                                  size_t input_index) {
  // NOTE: Parameters are initialized on executor place defined
  if ((op == pir::SetParameterOp::name() ||
       op == pir::ShadowOutputOp::name()) &&
      phi::is_accelerat_allocation_type(place.GetType())) {
    return phi::TransToPhiBackend(place);
  }
  // Tensor Attribute should on cpu backend for better performance
  if (op_info_parser != nullptr &&
      op_info_parser->IsTensorAttribute(input_index)) {
    return phi::Backend::CPU;
  }
  return kernel_backend;
}

static phi::Backend ChooseInputBackend(const phi::Kernel& kernel,
                                       size_t input_index,
                                       phi::Backend default_backend) {
  if (kernel.GetKernelRegisteredType() == phi::KernelRegisteredType::FUNCTION) {
    return kernel.InputAt(input_index).backend;
  }
  return default_backend;
}

static std::set<std::string> GetInputsByDataOp(pir::Block* block) {
  std::set<std::string> data_op_names;
  for (auto& op_item : *block) {
    if (op_item.isa<DataOp>()) {
      data_op_names.insert(op_item.attributes()
                               .at("name")
                               .dyn_cast<pir::StrAttribute>()
                               .AsString());
    }
  }
  return data_op_names;
}

template <class IrType>
static phi::DenseTensorMeta parse_tensor_meta(IrType type) {
  auto dtype = TransToPhiDataType(type.dtype());
  return phi::DenseTensorMeta(
      dtype, type.dims(), type.data_layout(), type.lod(), type.offset());
}

template <>
phi::DenseTensorMeta parse_tensor_meta<AllocatedSparseCooTensorType>(
    AllocatedSparseCooTensorType type) {
  auto dtype = TransToPhiDataType(type.dtype());
  return phi::DenseTensorMeta(dtype, type.dims(), type.data_layout());
}

template <>
phi::DenseTensorMeta parse_tensor_meta<AllocatedSparseCsrTensorType>(
    AllocatedSparseCsrTensorType type) {
  auto dtype = TransToPhiDataType(type.dtype());
  return phi::DenseTensorMeta(dtype, type.dims(), type.data_layout());
}

static std::vector<std::shared_ptr<phi::TensorBase>> PrepareFakeTensors(
    pir::Value input) {
  std::vector<std::shared_ptr<phi::TensorBase>> res;
  auto in_type = input.type();

  auto fake_dt = [](const AllocatedDenseTensorType& type) {
    auto ptr = new phi::Allocation(nullptr, 0, type.place());
    std::shared_ptr<phi::Allocation> holder(ptr);
    phi::DenseTensorMeta meta =
        parse_tensor_meta<AllocatedDenseTensorType>(type);
    return std::make_shared<phi::DenseTensor>(holder, meta);
  };

  auto fake_spcoo = [](const AllocatedSparseCooTensorType& type) {
    auto ptr = new phi::Allocation(nullptr, 0, type.place());
    std::shared_ptr<phi::Allocation> holder(ptr);
    phi::DenseTensorMeta meta =
        parse_tensor_meta<AllocatedSparseCooTensorType>(type);
    return std::make_shared<phi::DenseTensor>(holder, meta);
  };

  auto fake_spcsr = [](const AllocatedSparseCsrTensorType& type) {
    auto ptr = new phi::Allocation(nullptr, 0, type.place());
    std::shared_ptr<phi::Allocation> holder(ptr);
    phi::DenseTensorMeta meta =
        parse_tensor_meta<AllocatedSparseCsrTensorType>(type);

    return std::make_shared<phi::DenseTensor>(holder, meta);
  };
  auto fake_sr = [](const AllocatedSelectedRowsType& type) {
    auto ptr = new phi::Allocation(nullptr, 0, type.place());
    std::shared_ptr<phi::Allocation> holder(ptr);
    phi::DenseTensorMeta meta =
        parse_tensor_meta<AllocatedSelectedRowsType>(type);

    std::vector<int64_t> rows;
    rows.clear();
    auto sr = std::make_shared<phi::SelectedRows>(rows, 0);
    phi::DenseTensor dense_tensor(holder, meta);
    *(sr->mutable_value()) = dense_tensor;

    return sr;
  };

  auto fake_tensor_array = [](const AllocatedDenseTensorArrayType& type) {
    auto ptr = new phi::Allocation(nullptr, 0, type.place());
    std::shared_ptr<phi::Allocation> holder(ptr);
    auto dtype = TransToPhiDataType(type.dtype());
    phi::DenseTensorMeta meta(dtype, {});
    phi::DenseTensor dt(holder, meta);
    auto tensor_array = std::make_shared<phi::TensorArray>(0);
    tensor_array->set_type(dtype);
    return tensor_array;
  };

  if (in_type.isa<AllocatedDenseTensorType>()) {
    res.push_back(fake_dt(in_type.dyn_cast<AllocatedDenseTensorType>()));
  } else if (in_type.isa<AllocatedSelectedRowsType>()) {
    res.push_back(fake_sr(in_type.dyn_cast<AllocatedSelectedRowsType>()));
  } else if (in_type.isa<AllocatedSparseCsrTensorType>()) {
    res.push_back(fake_spcsr(in_type.dyn_cast<AllocatedSparseCsrTensorType>()));
  } else if (in_type.isa<AllocatedSparseCooTensorType>()) {
    res.push_back(fake_spcoo(in_type.dyn_cast<AllocatedSparseCooTensorType>()));
  } else if (in_type.isa<pir::VectorType>()) {
    auto inner_types = in_type.dyn_cast<pir::VectorType>().data();
    for (size_t i = 0; i < inner_types.size(); ++i) {
      if (inner_types[i].isa<AllocatedDenseTensorType>()) {
        res.push_back(
            fake_dt(inner_types[i].dyn_cast<AllocatedDenseTensorType>()));
      } else if (inner_types[i].isa<AllocatedSelectedRowsType>()) {
        res.push_back(
            fake_sr(inner_types[i].dyn_cast<AllocatedSelectedRowsType>()));
      } else if (inner_types[i].isa<AllocatedDenseTensorArrayType>()) {
        res.push_back(fake_tensor_array(
            inner_types[i].dyn_cast<AllocatedDenseTensorArrayType>()));
      }
    }
  } else if (in_type.isa<AllocatedDenseTensorArrayType>()) {
    res.push_back(
        fake_tensor_array(in_type.dyn_cast<AllocatedDenseTensorArrayType>()));
  }

  return res;
}

static pir::Value AddPlaceTransferOp(pir::Value in,
                                     pir::Type out_type,
                                     const phi::Place& src_place,
                                     const phi::Place& dst_place,
                                     const phi::KernelKey& kernel_key,
                                     pir::Block* block) {
  pir::IrContext* ctx = pir::IrContext::Instance();

  auto copy_kernel_key = kernel_key;
  auto place2backend = [](phi::AllocationType new_place_type) {
    auto new_backend = phi::Backend::GPU;
    switch (new_place_type) {
      case phi::AllocationType::GPU:
        new_backend = phi::Backend::GPU;
        break;
      case phi::AllocationType::XPU:
        new_backend = phi::Backend::XPU;
        break;
      case phi::AllocationType::CUSTOM:
        new_backend = phi::Backend::CUSTOM;
        break;
      case phi::AllocationType::IPU:
        new_backend = phi::Backend::IPU;
        break;
      default:
        new_backend = phi::Backend::CPU;
        break;
    }
    return new_backend;
  };
  std::unordered_map<std::string, pir::Attribute> op_attribute;
  if ((src_place.GetType() == phi::AllocationType::CPU) &&
      phi::is_accelerat_allocation_type(dst_place.GetType())) {
    copy_kernel_key.set_backend(place2backend(dst_place.GetType()));
    op_attribute = {
        {"op_name", pir::StrAttribute::get(ctx, "pd_op.memcpy_h2d")},
        {"kernel_name", pir::StrAttribute::get(ctx, "memcpy_h2d")},
        {"kernel_key", KernelAttribute::get(ctx, copy_kernel_key)},
        {"dst_place_type", pir::Int32Attribute::get(ctx, 1)}};
  } else if (phi::is_accelerat_allocation_type(src_place.GetType()) &&
             (dst_place.GetType() == phi::AllocationType::CPU)) {
    copy_kernel_key.set_backend(place2backend(src_place.GetType()));

    std::string copy_kernel_name = "memcpy_d2h";
    if (in.type().isa<AllocatedDenseTensorArrayType>()) {
      copy_kernel_name = "memcpy_d2h_multi_io";
    }
    op_attribute = {
        {"op_name", pir::StrAttribute::get(ctx, "pd_op." + copy_kernel_name)},
        {"kernel_name", pir::StrAttribute::get(ctx, copy_kernel_name)},
        {"kernel_key", KernelAttribute::get(ctx, copy_kernel_key)},
        {"dst_place_type", pir::Int32Attribute::get(ctx, 0)}};
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support cpu to gpu and gpu to cpu, src=%s, dst=%s.",
        src_place,
        dst_place));
  }

  pir::OpInfo kernel_op_info = ctx->GetRegisteredOpInfo(PhiKernelOp::name());
  pir::Operation* op =
      pir::Operation::Create({in}, op_attribute, {out_type}, kernel_op_info);
  op->set_attribute("origin_id", pir::Int64Attribute::get(ctx, op->id()));

  auto in_op = in.defining_op();
  if (in_op && in_op->HasAttribute(kAttrIsPersistable)) {
    op->set_attribute(kAttrIsPersistable, in_op->attribute(kAttrIsPersistable));
  }
  block->push_back(op);
  auto new_in = op->result(0);
  return new_in;
}

#ifdef PADDLE_WITH_DNNL
static pir::Value AddOneDNN2PaddleLayoutTransferOp(
    pir::Value in, const phi::DataLayout& dst_layout, pir::Block* block) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  auto in_alloc_type = in.type().dyn_cast<AllocatedDenseTensorType>();

  phi::KernelKey kernel_key;
  kernel_key.set_backend(phi::Backend::CPU);
  kernel_key.set_layout(phi::DataLayout::ANY);
  kernel_key.set_dtype(dialect::TransToPhiDataType(in_alloc_type.dtype()));

  std::unordered_map<std::string, pir::Attribute> op_attribute;
  op_attribute = {
      {"op_name", pir::StrAttribute::get(ctx, "pd_op.onednn_to_paddle_layout")},
      {"kernel_name", pir::StrAttribute::get(ctx, "onednn_to_paddle_layout")},
      {"kernel_key", KernelAttribute::get(ctx, kernel_key)},
      {"dst_layout",
       pir::Int32Attribute::get(ctx, static_cast<int>(dst_layout))}};

  auto out_type = AllocatedDenseTensorType::get(ctx,
                                                in_alloc_type.place(),
                                                in_alloc_type.dtype(),
                                                in_alloc_type.dims(),
                                                dst_layout,
                                                in_alloc_type.lod(),
                                                in_alloc_type.offset());

  pir::OpInfo kernel_op_info = ctx->GetRegisteredOpInfo(PhiKernelOp::name());
  pir::Operation* op =
      pir::Operation::Create({in}, op_attribute, {out_type}, kernel_op_info);
  op->set_attribute("origin_id", pir::Int64Attribute::get(ctx, op->id()));

  auto in_op = in.defining_op();
  if (in_op && in_op->HasAttribute(kAttrIsPersistable)) {
    op->set_attribute(kAttrIsPersistable, in_op->attribute(kAttrIsPersistable));
  }

  block->push_back(op);
  return op->result(0);
}
#endif

static bool NeedTransformDataType(const phi::DataType& l,
                                  const phi::DataType& r) {
  return l != phi::DataType::ALL_DTYPE && r != phi::DataType::ALL_DTYPE &&
         l != r;
}

static const phi::DataType GetKernelTypeforVar(
    pir::Operation* op,
    const std::string& var_name,
    const phi::DataType& tensor_dtype,
    const phi::KernelKey* expected_kernel_key) {
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op->name());
  auto get_kernel_type_for_var =
      op_info.GetInterfaceImpl<GetKernelTypeForVarInterface>();
  if (get_kernel_type_for_var) {
    phi::DataType kernel_dtype_for_var =
        get_kernel_type_for_var->get_kernel_type_for_var_(
            var_name, tensor_dtype, (*expected_kernel_key).dtype());
    return kernel_dtype_for_var;
  }
  return (*expected_kernel_key).dtype();
}

template <class IrType>
std::tuple<phi::Backend, phi::DataLayout> parse_kernel_info(pir::Type type) {
  phi::Backend backend =
      paddle::experimental::ParseBackend(type.dyn_cast<IrType>().place());
  phi::DataLayout layout =
      paddle::experimental::ParseLayout(type.dyn_cast<IrType>().data_layout());
  return {backend, layout};
}

template <class IrType1, class IrType2>
static pir::Type create_sparse_coo_tensor_type(pir::Type type,
                                               const phi::Place& place,
                                               pir::Type out_dtype,
                                               pir::IrContext* ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      input_type.non_zero_dims(),
                      input_type.data_layout(),
                      input_type.non_zero_indices(),
                      input_type.non_zero_elements(),
                      input_type.coalesced());
}

template <class IrType1, class IrType2>
static pir::Type create_sparse_csr_tensor_type(pir::Type type,
                                               const phi::Place& place,
                                               pir::Type out_dtype,
                                               pir::IrContext* ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      input_type.data_layout(),
                      input_type.non_zero_crows(),
                      input_type.non_zero_cols(),
                      input_type.non_zero_elements());
}

template <class IrType1, class IrType2>
static pir::Type create_type(pir::Type type,
                             const phi::Place& place,
                             pir::Type out_dtype,
                             pir::IrContext* ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      input_type.data_layout(),
                      input_type.lod(),
                      input_type.offset());
}

static pir::Type BuildDtypeTransferOutputType(pir::Type type,
                                              const phi::Place& place,
                                              phi::DataType data_dtype,
                                              pir::IrContext* ctx) {
  if (type.isa<AllocatedDenseTensorType>()) {
    auto out_dtype = TransToIrDataType(data_dtype, ctx);
    return create_type<AllocatedDenseTensorType, AllocatedDenseTensorType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<AllocatedSelectedRowsType>()) {
    auto out_dtype = TransToIrDataType(data_dtype, ctx);
    return create_type<AllocatedSelectedRowsType, AllocatedSelectedRowsType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<AllocatedSparseCooTensorType>()) {
    auto out_dtype = TransToIrDataType(data_dtype, ctx);
    return create_sparse_coo_tensor_type<AllocatedSparseCooTensorType,
                                         AllocatedSparseCooTensorType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<AllocatedSparseCsrTensorType>()) {
    auto out_dtype = TransToIrDataType(data_dtype, ctx);
    return create_sparse_csr_tensor_type<AllocatedSparseCsrTensorType,
                                         AllocatedSparseCsrTensorType>(
        type, place, out_dtype, ctx);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "BuildOutputType only support DenseTensorType, SelectedRowsType, "
        "SparseCooTensorType and SparseCsrTensorType"));
  }
}

static pir::Type BuildOutputType(pir::Type type,
                                 const phi::Place& place,
                                 pir::IrContext* ctx) {
  if (type.isa<DenseTensorType>()) {
    auto out_dtype = type.dyn_cast<DenseTensorType>().dtype();
    return create_type<DenseTensorType, AllocatedDenseTensorType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<SelectedRowsType>()) {
    auto out_dtype = type.dyn_cast<SelectedRowsType>().dtype();
    return create_type<SelectedRowsType, AllocatedSelectedRowsType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<DenseTensorArrayType>()) {
    auto array_type = type.dyn_cast<DenseTensorArrayType>();
    return AllocatedDenseTensorArrayType::get(ctx,
                                              place,
                                              array_type.dtype(),
                                              array_type.dims(),
                                              array_type.data_layout());
  } else if (type.isa<SparseCooTensorType>()) {
    auto out_dtype = type.dyn_cast<SparseCooTensorType>().dtype();
    return create_sparse_coo_tensor_type<SparseCooTensorType,
                                         AllocatedSparseCooTensorType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<SparseCsrTensorType>()) {
    auto out_dtype = type.dyn_cast<SparseCsrTensorType>().dtype();
    return create_sparse_csr_tensor_type<SparseCsrTensorType,
                                         AllocatedSparseCsrTensorType>(
        type, place, out_dtype, ctx);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "BuildOutputType only support DenseTensorType, SelectedRowsType, "
        "SparseCooTensorType and SparseCsrTensorType"));
  }
}

#ifdef PADDLE_WITH_DNNL
template <class IrType1, class IrType2>
static pir::Type create_type(pir::Type type,
                             const phi::Place& place,
                             const phi::DataLayout& layout,
                             pir::Type out_dtype,
                             pir::IrContext* ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      layout,
                      input_type.lod(),
                      input_type.offset());
}

static pir::Type BuildOutputType(pir::Type type,
                                 const phi::Place& place,
                                 const phi::DataLayout& layout,
                                 pir::IrContext* ctx) {
  if (type.isa<DenseTensorType>()) {
    auto out_dtype = type.dyn_cast<DenseTensorType>().dtype();
    return create_type<DenseTensorType, AllocatedDenseTensorType>(
        type, place, layout, out_dtype, ctx);
  } else if (type.isa<SelectedRowsType>()) {
    auto out_dtype = type.dyn_cast<SelectedRowsType>().dtype();
    return create_type<SelectedRowsType, AllocatedSelectedRowsType>(
        type, place, layout, out_dtype, ctx);
  } else if (type.isa<DenseTensorArrayType>()) {
    auto array_type = type.dyn_cast<DenseTensorArrayType>();
    return AllocatedDenseTensorArrayType::get(
        ctx, place, array_type.dtype(), array_type.dims(), layout);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "BuildOutputType only support DenseTensorType and SelectedRowsType"));
  }
}
#endif

pir::Value AddDtypeTransferOp(pir::Value in,
                              pir::Block* block,
                              const phi::KernelKey& kernel_key,
                              const phi::Place& origin_place,
                              const phi::Place& out_place,
                              const phi::DataType& src_dtype,
                              const phi::DataType& dst_dtype) {
  pir::IrContext* ctx = pir::IrContext::Instance();

  pir::OpInfo kernel_op_info = ctx->GetRegisteredOpInfo(PhiKernelOp::name());

  // Get kernelkey (backend、layout)
  phi::Backend kernel_backend = phi::Backend::UNDEFINED;
  phi::DataLayout kernel_layout = phi::DataLayout::UNDEFINED;

  if (in.type().isa<AllocatedDenseTensorType>()) {
    auto out = parse_kernel_info<AllocatedDenseTensorType>(in.type());
    kernel_backend = std::get<0>(out);
    kernel_layout = std::get<1>(out);
  } else if (in.type().isa<AllocatedSelectedRowsType>()) {
    auto out = parse_kernel_info<AllocatedSelectedRowsType>(in.type());
    kernel_backend = std::get<0>(out);
    kernel_layout = std::get<1>(out);
  } else if (in.type().isa<AllocatedSparseCooTensorType>()) {
    auto out = parse_kernel_info<AllocatedSparseCooTensorType>(in.type());
    kernel_backend = std::get<0>(out);
    kernel_layout = std::get<1>(out);
  } else if (in.type().isa<AllocatedSparseCsrTensorType>()) {
    auto out = parse_kernel_info<AllocatedSparseCsrTensorType>(in.type());
    kernel_backend = std::get<0>(out);
    kernel_layout = std::get<1>(out);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Get kernelkey for CastOp only support "
        "DenseTensorType, SparseCooTensorType, SparseCsrTensorType, and "
        "SelectedRowsType"));
  }

  if (kernel_backend == phi::Backend::UNDEFINED) {
    kernel_backend = paddle::experimental::ParseBackend(origin_place);
  }

  phi::KernelKey cast_kernel_key(kernel_backend, kernel_layout, src_dtype);

  // Create CastOp
  std::unordered_map<std::string, pir::Attribute> op_attribute{
      {"op_name", pir::StrAttribute::get(ctx, "pd_op.cast")},
      {"kernel_name", pir::StrAttribute::get(ctx, "cast")},
      {"kernel_key", KernelAttribute::get(ctx, cast_kernel_key)},
      {"dtype", DataTypeAttribute::get(ctx, dst_dtype)}};

  pir::Type output_types =
      BuildDtypeTransferOutputType(in.type(), out_place, dst_dtype, ctx);

  pir::Operation* op = pir::Operation::Create(
      {in}, op_attribute, {output_types}, kernel_op_info);
  op->set_attribute("origin_id", pir::Int64Attribute::get(ctx, op->id()));

  auto in_op = in.defining_op();
  if (in_op && in_op->HasAttribute(kAttrIsPersistable)) {
    op->set_attribute(kAttrIsPersistable, in_op->attribute(kAttrIsPersistable));
  }
  block->push_back(op);
  pir::Value new_in = op->result(0);
  return new_in;
}

static phi::DataType GetKernelDtypeByYaml(
    const pir::Operation* op,
    const std::unordered_map<pir::Value, pir::Value>& map_value_pair,
    const OpYamlInfoParser* op_info_parser) {
  auto& attr_map = op->attributes();
  auto& data_type_info = op_info_parser->OpRuntimeInfo().kernel_key_dtype;
  phi::DataType kernel_data_type = phi::DataType::UNDEFINED;

  for (auto slot_name : data_type_info) {
    auto& input_map = op_info_parser->InputName2Id();

    bool is_complex_tag = false;
    if (slot_name.find("complex:") == 0) {
      slot_name = slot_name.substr(8);
      is_complex_tag = true;
    }

    auto find_it = Str2PhiDataType.find(slot_name);
    if (find_it != Str2PhiDataType.end()) {
      kernel_data_type = find_it->second;
    } else if (input_map.count(slot_name)) {
      // parse from input
      int in_index = static_cast<int>(input_map.at(slot_name));
      auto type = map_value_pair.at(op->operand_source(in_index)).type();
      if (type.isa<AllocatedDenseTensorType>()) {
        kernel_data_type = TransToPhiDataType(
            type.dyn_cast<AllocatedDenseTensorType>().dtype());
      } else if (type.isa<pir::VectorType>()) {
        auto vec_data = type.dyn_cast<pir::VectorType>().data();
        if (vec_data.empty()) {
          kernel_data_type = phi::DataType::UNDEFINED;
        } else {
          if (vec_data[0].isa<AllocatedDenseTensorType>()) {
            kernel_data_type = TransToPhiDataType(
                vec_data[0].dyn_cast<AllocatedDenseTensorType>().dtype());
          } else if (vec_data[0].isa<AllocatedSelectedRowsType>()) {
            kernel_data_type = TransToPhiDataType(
                vec_data[0].dyn_cast<AllocatedSelectedRowsType>().dtype());
          } else if (vec_data[0].isa<AllocatedSparseCooTensorType>()) {
            kernel_data_type = TransToPhiDataType(
                vec_data[0].dyn_cast<AllocatedSparseCooTensorType>().dtype());
          } else if (vec_data[0].isa<AllocatedSparseCsrTensorType>()) {
            kernel_data_type = TransToPhiDataType(
                vec_data[0].dyn_cast<AllocatedSparseCsrTensorType>().dtype());
          } else {
            PADDLE_THROW(common::errors::Unimplemented(
                "Only support DenseTensorType and SelectedRowsType in vector"));
          }
        }
      } else if (type.isa<AllocatedSelectedRowsType>()) {
        kernel_data_type = TransToPhiDataType(
            type.dyn_cast<AllocatedSelectedRowsType>().dtype());
      } else if (type.isa<AllocatedSparseCooTensorType>()) {
        kernel_data_type = TransToPhiDataType(
            type.dyn_cast<AllocatedSparseCooTensorType>().dtype());
      } else if (type.isa<AllocatedSparseCsrTensorType>()) {
        kernel_data_type = TransToPhiDataType(
            type.dyn_cast<AllocatedSparseCsrTensorType>().dtype());
      } else if (type.isa<AllocatedDenseTensorArrayType>()) {
        kernel_data_type = TransToPhiDataType(
            type.dyn_cast<AllocatedDenseTensorArrayType>().dtype());
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Only support DenseTensorType, SelectedRows, SparseCooTensorType, "
            "SparseCsrTensorType and VectorType"));
      }
      if (is_complex_tag) {
        kernel_data_type = phi::dtype::ToComplex(kernel_data_type);
      }

    } else {
      PADDLE_ENFORCE_EQ(attr_map.count(slot_name),
                        true,
                        common::errors::PreconditionNotMet(
                            "[%s] MUST in attribute map", slot_name));

      auto attr_type = op_info_parser->AttrTypeName(slot_name);
      PADDLE_ENFORCE_EQ(attr_type,
                        "paddle::dialect::DataTypeAttribute",
                        common::errors::PreconditionNotMet(
                            "Type of [%s] should be DataType", slot_name));
      kernel_data_type =
          attr_map.at(slot_name).dyn_cast<DataTypeAttribute>().data();
    }

    if (kernel_data_type != phi::DataType::UNDEFINED) {
      // In yaml definition, data type have an order
      // like: data_type : dtype > x
      // Should break when found a defined data type
      break;
    }
  }
  return kernel_data_type;
}

static phi::Backend GetKernelBackendByYaml(
    const pir::Operation* op,
    const std::unordered_map<pir::Value, pir::Value>& map_value_pair,
    const OpYamlInfoParser* op_info_parser,
    const phi::Place& place) {
  auto& attr_map = op->attributes();
  auto& backend_info = op_info_parser->OpRuntimeInfo().kernel_key_backend;
  phi::Backend kernel_backend = phi::Backend::UNDEFINED;

  for (const auto& slot_name : backend_info) {
    auto& input_map = op_info_parser->InputName2Id();
    if (input_map.count(slot_name)) {
      // parse from input
      int in_index = static_cast<int>(input_map.at(slot_name));
      auto type = map_value_pair.at(op->operand_source(in_index)).type();
      if (type.isa<AllocatedDenseTensorType>()) {
        kernel_backend = paddle::experimental::ParseBackend(
            type.dyn_cast<AllocatedDenseTensorType>().place());
      } else if (type.isa<pir::VectorType>()) {
        auto vec_data = type.dyn_cast<pir::VectorType>().data();
        if (vec_data.empty()) {
          kernel_backend = phi::Backend::UNDEFINED;
        } else {
          if (vec_data[0].isa<AllocatedDenseTensorType>()) {
            kernel_backend = paddle::experimental::ParseBackend(
                vec_data[0].dyn_cast<AllocatedDenseTensorType>().place());
          } else {
            PADDLE_THROW(common::errors::Unimplemented(
                "Only support DenseTensorType in vector"));
          }
        }
      } else if (type.isa<AllocatedSelectedRowsType>()) {
        kernel_backend = paddle::experimental::ParseBackend(
            type.dyn_cast<AllocatedSelectedRowsType>().place());
      } else if (type.isa<AllocatedSparseCooTensorType>()) {
        kernel_backend = paddle::experimental::ParseBackend(
            type.dyn_cast<AllocatedSparseCooTensorType>().place());
      } else if (type.isa<AllocatedSparseCsrTensorType>()) {
        kernel_backend = paddle::experimental::ParseBackend(
            type.dyn_cast<AllocatedSparseCsrTensorType>().place());
      } else if (type.isa<AllocatedDenseTensorArrayType>()) {
        kernel_backend = paddle::experimental::ParseBackend(
            type.dyn_cast<AllocatedDenseTensorArrayType>().place());
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Only support DenseTensorType, SelectedRows, SparseCooTensorType, "
            "SparseCsrTensorType and VectorType"));
      }
    } else {
      PADDLE_ENFORCE_EQ(attr_map.count(slot_name),
                        true,
                        common::errors::PreconditionNotMet(
                            "[%s] MUST in attribute map", slot_name));

      auto attr_type = op_info_parser->AttrTypeName(slot_name);
      PADDLE_ENFORCE_EQ(attr_type,
                        "paddle::dialect::PlaceAttribute",
                        common::errors::PreconditionNotMet(
                            "Type of [%s] should be DataType", slot_name));
      kernel_backend = paddle::experimental::ParseBackend(
          attr_map.at(slot_name).dyn_cast<PlaceAttribute>().data());
    }
    if (kernel_backend != phi::Backend::UNDEFINED) {
      // In yaml definition, backend have an order
      // like: backend : place > x
      // Should break when found a defined data type
      break;
    }
  }

  if (!backend_info.empty() && kernel_backend == phi::Backend::UNDEFINED) {
    kernel_backend = paddle::experimental::ParseBackend(place);
  }
  return kernel_backend;
}

std::unique_ptr<OpYamlInfoParser> GetOpYamlInfoParser(pir::Operation* op) {
  OpYamlInfoInterface op_info_interface = op->dyn_cast<OpYamlInfoInterface>();

  std::unique_ptr<OpYamlInfoParser> op_info_parser(nullptr);
  if (op_info_interface) {
    op_info_parser = std::make_unique<OpYamlInfoParser>(
        op_info_interface.GetOpInfo(), IsLegacyOp(op->name()));
  }

  return op_info_parser;
}

std::string GetKernelName(const OpYamlInfoParser* op_info_parser,
                          pir::Operation* op_item) {
  std::string kernel_fn_str;
  if (op_info_parser != nullptr) {
    kernel_fn_str = op_info_parser->OpRuntimeInfo().kernel_func;
  }

  if (op_item->isa<AddN_Op>() || op_item->isa<AddNOp>()) {
    if (op_item->result(0).type().isa<SelectedRowsType>()) {
      kernel_fn_str = "add_n_sr";
    }
  }
  if (op_item->isa<FetchOp>()) {
    if (op_item->result(0).type().isa<DenseTensorArrayType>()) {
      kernel_fn_str = "fetch_array";
    }
  }
  return kernel_fn_str;
}

#ifdef PADDLE_WITH_DNNL
bool SupportsMKLDNN(const std::string& kernel_name,
                    const phi::DataType data_type) {
  auto phi_kernels =
      phi::KernelFactory::Instance().SelectKernelMap(kernel_name);
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
    auto op_kernel_iter =
        paddle::framework::OperatorWithKernel::AllOpKernels().find(
            phi::TransToFluidOpName(kernel_name));
    if (op_kernel_iter ==
        paddle::framework::OperatorWithKernel::AllOpKernels().end()) {
      return false;
    } else {
      auto& op_kernels = op_kernel_iter->second;
      return std::any_of(
          op_kernels.begin(),
          op_kernels.end(),
          [data_type](std::unordered_map<
                      paddle::framework::OpKernelType,
                      std::function<void(
                          const paddle::framework::ExecutionContext&)>,
                      paddle::framework::OpKernelType::Hash>::const_reference
                          kern_pair) {
            return phi::is_cpu_place(kern_pair.first.place_) &&
                   kern_pair.first.library_type_ ==
                       paddle::framework::LibraryType::kMKLDNN &&
                   kern_pair.first.data_type_ ==
                       paddle::framework::TransToProtoVarType(data_type);
          });
    }
  }
}

bool SupportsCPUBF16(const std::string& kernel_name) {
  auto phi_kernels =
      phi::KernelFactory::Instance().SelectKernelMap(kernel_name);
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
    auto op_kernel_iter =
        paddle::framework::OperatorWithKernel::AllOpKernels().find(
            phi::TransToFluidOpName(kernel_name));
    if (op_kernel_iter ==
        paddle::framework::OperatorWithKernel::AllOpKernels().end()) {
      return false;
    } else {
      auto& op_kernels = op_kernel_iter->second;
      return std::any_of(
          op_kernels.begin(),
          op_kernels.end(),
          [](std::unordered_map<
              paddle::framework::OpKernelType,
              std::function<void(const paddle::framework::ExecutionContext&)>,
              paddle::framework::OpKernelType::Hash>::const_reference
                 kern_pair) {
            return phi::is_cpu_place(kern_pair.first.place_) &&
                   kern_pair.first.place_ == phi::CPUPlace() &&
                   kern_pair.first.data_type_ ==
                       paddle::framework::proto::VarType::Type::
                           VarType_Type_BF16;
          });
    }
  }
}
#endif

phi::KernelKey GetKernelKey(
    pir::Operation* op,
    const phi::Place& place,
    const std::string& kernel_fn_str,
    const std::unordered_map<pir::Value, pir::Value>& map_value_pair,
    OpYamlInfoParser* op_info_parser = nullptr) {
  if (op->isa<FeedOp>() || op->isa<FetchOp>() || op->isa<ArrayLengthOp>()) {
    // NOTE, for now feed op don't need a kernel, so the data type from Op
    // Result the next op use base program datatype
    VLOG(6) << "FeedOp doesn't need a kernel. Backend: CPU, DataLayout: ANY";
    pir::Type dtype;
    if (op->result(0).type().isa<paddle::dialect::DenseTensorArrayType>()) {
      dtype = op->result(0)
                  .type()
                  .dyn_cast<paddle::dialect::DenseTensorArrayType>()
                  .dtype();
    } else if (op->result(0).type().isa<paddle::dialect::DenseTensorType>()) {
      dtype = op->result(0)
                  .type()
                  .dyn_cast<paddle::dialect::DenseTensorType>()
                  .dtype();
    } else {
      PADDLE_THROW(
          "FeedOp, FetchOp, ArrayLengthOp can only output a densetensor or "
          "dense tensor array.");
    }
    return {phi::Backend::CPU, phi::DataLayout::ANY, TransToPhiDataType(dtype)};
  }

  if (op->isa<DataOp>()) {
    // NOTE, for now feed op don't need a kernel, so the data type from Op
    // Result the next op use base program datatype
    VLOG(6) << "DataOp doesn't need a kernel";
    auto data_place =
        op->attributes().at("place").dyn_cast<PlaceAttribute>().data();

    phi::Backend backend;
    if (data_place.GetType() == AllocationType::GPUPINNED) {
      backend = phi::Backend::CPU;
    } else {
      backend = paddle::experimental::ParseBackend(data_place);
    }

    return {backend,
            phi::DataLayout::ANY,
            TransToPhiDataType(
                op->result(0).type().dyn_cast<DenseTensorType>().dtype())};
  }

  if (op->isa<SeedOp>()) {
    VLOG(6) << "SeedOp doesn't need a kernel";
    auto backend = paddle::experimental::ParseBackend(place);
    return {backend, phi::DataLayout::ANY, phi::DataType::INT32};
  }

  if (op->isa<FullWithTensorOp>()) {
    VLOG(6) << "FullWithTensorOp doesn't need a kernel";
    auto backend = paddle::experimental::ParseBackend(place);
    auto dtype =
        op->attributes().at("dtype").dyn_cast<DataTypeAttribute>().data();

    return {backend, phi::DataLayout::ANY, dtype};
  }

  if (op->isa<CreateArrayOp>()) {
    VLOG(6) << "CreateArrayOp doesn't need a kernel";
    auto backend = paddle::experimental::ParseBackend(place);
    auto dtype = op->result(0)
                     .type()
                     .dyn_cast<paddle::dialect::DenseTensorArrayType>()
                     .dtype();

    phi::KernelKey res(
        backend, phi::DataLayout::ANY, TransToPhiDataType(dtype));
    if (NeedFallBackCpu(op, kernel_fn_str, res)) {
      res.set_backend(phi::Backend::CPU);
    }
    return res;
  }

  phi::Backend kernel_backend = phi::Backend::UNDEFINED;
  phi::DataLayout kernel_layout = phi::DataLayout::UNDEFINED;
  phi::DataType kernel_dtype = phi::DataType::UNDEFINED;

  if (op_info_parser != nullptr) {
    // only support non vector input for now
    int tensor_input_number =
        static_cast<int>(op_info_parser->InputTensorNumber());
    VLOG(8) << "Begin to infer kernel key from op_info_parser(defined by yaml "
               "info)";
    // get datatype info
    kernel_dtype = GetKernelDtypeByYaml(op, map_value_pair, op_info_parser);
    VLOG(8) << "Infer kernel data_type: [" << kernel_dtype
            << "] from yaml info";
    kernel_backend =
        GetKernelBackendByYaml(op, map_value_pair, op_info_parser, place);
    VLOG(8) << "Infer kernel backend: [" << kernel_backend
            << "] from yaml info";
    // parse all the input tensor
    if (tensor_input_number == 0 || op->isa<Full_Op>()) {
      // all the information have to get from attribute and context
      if (kernel_backend == phi::Backend::UNDEFINED) {
        kernel_backend = paddle::experimental::ParseBackend(place);
        VLOG(8) << "Infer kernel backend: [" << kernel_backend
                << "] when tensor_input_number == 0  or is Full_Op";
      }
    }
  }

  // TODO(zhangbo): Add ParseKernelInterface
  ParseKernelKeyInterface parse_kernel_key_interface =
      op->dyn_cast<ParseKernelKeyInterface>();
  if (parse_kernel_key_interface) {
    auto parsed_key = parse_kernel_key_interface.ParseKernelKey(op);
    kernel_dtype = std::get<0>(parsed_key);
    kernel_backend = std::get<1>(parsed_key);
  }

  if ((kernel_backend == phi::Backend::UNDEFINED ||
       kernel_dtype == phi::DataType::UNDEFINED) &&
      op->num_operands() > 0) {
    paddle::experimental::detail::KernelKeyParser kernel_key_parser;
    VLOG(8) << "Begin to infer kernel key from op operands";
    for (size_t i = 0; i < op->num_operands(); ++i) {
      // NOTE, only op with OpYamlInfo can have TensorArr
      if (op_info_parser != nullptr && op_info_parser->IsTensorAttribute(i)) {
        VLOG(8) << "input (" << i << ") doesn't have TensorArr";
        continue;
      }
      auto input_tmp = op->operand_source(i);
      // NOTE: if not input_tmp, it's an optional input
      if (!input_tmp) {
        VLOG(8) << "input (" << i << ") is NULL (optional input)";
        continue;
      }
      auto new_input_tmp = map_value_pair.at(input_tmp);
      auto fake_tensors = PrepareFakeTensors(new_input_tmp);
      for (auto& fake_tensor : fake_tensors) {
        kernel_key_parser.AssignKernelKeySet(*fake_tensor);
      }

      // Because we can't make sure the place when build data op
      // and the output place of data op is undefined. It means we
      // don't know how to select the kernel in the next of op that
      // uses data op output as inputs. So, we need set kernel backend
      // manually.
      auto op_res = input_tmp.dyn_cast<pir::OpResult>();
      if (!op_res) {
        continue;
      }
      if (op_res.owner()->isa<DataOp>()) {
        auto data_op = op->operand_source(i).defining_op();
        auto data_place = data_op->attribute<PlaceAttribute>("place").data();

        auto data_op_backend = paddle::experimental::ParseBackend(data_place);
        if (data_op_backend == phi::Backend::UNDEFINED) {
          data_op_backend = paddle::experimental::ParseBackend(place);
        }
        kernel_key_parser.key_set.backend_set =
            kernel_key_parser.key_set.backend_set |
            paddle::experimental::BackendSet(data_op_backend);
        VLOG(8) << "Update kernel backend set from owner op (DataOp): "
                << data_op_backend;
      } else if (op_res.owner()->isa<pir::CombineOp>()) {
        auto combine_op = op_res.owner();
        for (size_t j = 0; j < combine_op->num_operands(); ++j) {
          auto combine_op_res =
              combine_op->operand_source(j).dyn_cast<pir::OpResult>();
          if (!combine_op_res) {
            continue;
          }

          if (combine_op_res.owner()->isa<DataOp>()) {
            auto data_op = combine_op_res.owner();
            auto data_place =
                data_op->attribute<PlaceAttribute>("place").data();

            auto data_op_backend =
                paddle::experimental::ParseBackend(data_place);
            if (data_op_backend == phi::Backend::UNDEFINED) {
              data_op_backend = paddle::experimental::ParseBackend(place);
            }
            kernel_key_parser.key_set.backend_set =
                kernel_key_parser.key_set.backend_set |
                paddle::experimental::BackendSet(data_op_backend);
            VLOG(8) << "Update kernel backend set from owner op (CombineOp): "
                    << data_op_backend;
            break;
          }
        }
      }
    }

    auto kernel_key_set = kernel_key_parser.key_set;

    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

    if (kernel_backend == phi::Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
      if (kernel_backend != phi::Backend::UNDEFINED) {
        VLOG(8) << "Infer kernel backend from op operands";
      }
    }
    if (kernel_layout == phi::DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
      if (kernel_layout != phi::DataLayout::UNDEFINED) {
        VLOG(8) << "Infer kernel layout from op operands";
      }
    }
    if (kernel_dtype == phi::DataType::UNDEFINED) {
      kernel_dtype = kernel_key.dtype();
      if (kernel_dtype != phi::DataType::UNDEFINED) {
        VLOG(8) << "Infer kernel data_type from op operands";
      }
    }
  }

  if (kernel_backend == phi::Backend::UNDEFINED) {
    VLOG(8) << "Kernel backend cannot be inferred from op operands";
    kernel_backend = paddle::experimental::ParseBackend(place);
  }

#ifdef PADDLE_WITH_DNNL
  if (kernel_backend != phi::Backend::ONEDNN &&
      kernel_layout == phi::DataLayout::ONEDNN) {
    kernel_layout = phi::DataLayout::ANY;
  }
#endif
  phi::KernelKey res(kernel_backend, kernel_layout, kernel_dtype);

  // kernel backend inferred incorrectly from memcpy op operands,
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // case that place from (not GPU) to GPU.
  // We handle this special case by following code to fix up the problem.
  // This could be further improved if we had another method.
  if (!phi::is_accelerat_place(place)) {
    if (op->isa<MemcpyOp>()) {
      VLOG(6) << "MemcpyOp need a special handle";
      int dst_place_type = op->attribute("dst_place_type")
                               .dyn_cast<pir::Int32Attribute>()
                               .data();
      if (dst_place_type == 1) {
        res.set_backend(phi::Backend::GPU);
      }
    }
  }
#endif

  if (op->isa<LoadCombineOp>()) {
    res.set_dtype(phi::DataType::FLOAT32);
    VLOG(8) << "LoadCombineOp's kernel data type must be FLOAT32";
  }

  if (op->isa<SyncCommStream_Op>() || op->isa<SyncCommStreamOp>()) {
    res.set_dtype(phi::DataType::FLOAT32);
    VLOG(8) << "SyncCommStream_Op/SyncCommStreamOp's kernel data type must "
               "be FLOAT32";
  }

  if (NeedFallBackCpu((op), kernel_fn_str, res)) {
    res.set_backend(phi::Backend::CPU);
#ifdef PADDLE_WITH_DNNL
    if (res.layout() == phi::DataLayout::ONEDNN) {
      res.set_layout(phi::DataLayout::ANY);
    }
#endif
    VLOG(8) << "kernel backend must be on CPU when need fallback";
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (NeedFallBackFromGPUDNN2GPU(op, kernel_fn_str, res)) {
    res.set_backend(phi::Backend::GPU);
    VLOG(8) << "kernel backend must be on GPU when need fallback from GPUDNN "
               "to GPU";
  }
#endif

#ifdef PADDLE_WITH_DNNL
  std::regex reg(",");
  std::string FLAGS_pir_onednn_kernel_blacklist;
  std::unordered_set<std::string> elems{
      std::sregex_token_iterator(FLAGS_pir_onednn_kernel_blacklist.begin(),
                                 FLAGS_pir_onednn_kernel_blacklist.end(),
                                 reg,
                                 -1),
      std::sregex_token_iterator()};
  elems.erase("");

  if (op->HasTrait<OneDNNTrait>() && res.backend() == phi::Backend::CPU &&
      SupportsMKLDNN(kernel_fn_str, res.dtype()) &&
      elems.count(op->name().substr(
          strlen(OneDNNOperatorDialect::name()) + 1,
          op->name().size() - strlen(OneDNNOperatorDialect::name()) - 1)) ==
          0) {
    res.set_backend(phi::Backend::ONEDNN);
    res.set_layout(phi::DataLayout::ONEDNN);
  }
#endif
  return res;
}

void HandleForIfOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  auto old_cond = op_item->operand_source(0);

  PADDLE_ENFORCE_EQ(
      map_value_pair->count(old_cond),
      true,
      common::errors::PreconditionNotMet(
          "[%d]'s input of [%s] op MUST in map pair", 0, op_item->name()));
  auto new_cond = map_value_pair->at(old_cond);

  // Create IfOp and insert to kernel dialect program
  pir::Builder builder(ctx, block);
  auto old_ifop = op_item->dyn_cast<IfOp>();
  std::vector<pir::Type> new_ifop_outputs;
  for (size_t i = 0; i < old_ifop.num_results(); ++i) {
    new_ifop_outputs.push_back(
        ConvertOpTypeToKernelType(ctx, old_ifop.result(i).type(), place));
  }
  auto new_ifop = builder.Build<IfOp>(new_cond, std::move(new_ifop_outputs));

  if (op_item->HasAttribute("fake_false_branch") &&
      op_item->attributes()
          .at("fake_false_branch")
          .dyn_cast<pir::BoolAttribute>()
          .data()) {
    new_ifop->set_attribute("fake_false_branch",
                            op_item->attribute("fake_false_branch"));
  }

  // process true block
  auto& true_block = new_ifop.true_block();
  ProcessBlock(place,
               &old_ifop.true_block(),
               &true_block,
               ctx,
               map_op_pair,
               map_value_pair,
               true);

  // process false block
  auto& false_block = new_ifop.false_block();
  ProcessBlock(place,
               &old_ifop.false_block(),
               &false_block,
               ctx,
               map_op_pair,
               map_value_pair,
               true);

  // update map
  (*map_op_pair)[op_item] = new_ifop;
  for (size_t i = 0; i < op_item->num_results(); ++i) {
    (*map_value_pair)[op_item->result(i)] = new_ifop->result(i);
  }
}

void HandleForPyLayerOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  std::vector<pir::Value> new_vec_input(op_item->num_operands());
  for (size_t index = 0; index < op_item->num_operands(); ++index) {
    const auto old_input = op_item->operand_source(index);

    PADDLE_ENFORCE_EQ(
        map_value_pair->count(old_input),
        true,
        common::errors::PreconditionNotMet(
            "[%d]'s input of [%s] op MUST in map pair", 0, op_item->name()));
    const auto& new_input = map_value_pair->at(old_input);
    new_vec_input[index] = new_input;
  }

  auto old_pylayerop = op_item->dyn_cast<PyLayerOp>();
  std::vector<pir::Type> new_pylayerop_outputs;
  for (size_t i = 0; i < old_pylayerop.num_results(); ++i) {
    if (!static_cast<bool>(old_pylayerop.result(i).type())) {
      new_pylayerop_outputs.push_back(old_pylayerop.result(i).type());
    } else {
      new_pylayerop_outputs.push_back(ConvertOpTypeToKernelType(
          ctx, old_pylayerop.result(i).type(), place));
    }
  }

  // Create PyLayerOp and insert to kernel dialect program
  pir::Builder builder(ctx, block);
  auto new_pylayerop =
      builder.Build<PyLayerOp>(new_vec_input,
                               std::move(new_pylayerop_outputs),
                               old_pylayerop.backward_function_id());

  // process sub block
  auto& fwd_block = new_pylayerop.forward_block();
  ProcessBlock(place,
               &old_pylayerop.forward_block(),
               &fwd_block,
               ctx,
               map_op_pair,
               map_value_pair,
               true);

  // update map
  (*map_op_pair)[op_item] = new_pylayerop;
  for (size_t i = 0; i < op_item->num_results(); ++i) {
    (*map_value_pair)[op_item->result(i)] = new_pylayerop->result(i);
  }
}

void HandleForWhileOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  std::vector<pir::Value> vec_in;
  pir::Value cond_val;
  for (size_t i = 0; i < op_item->num_operands(); ++i) {
    auto cur_in = op_item->operand_source(i);

    PADDLE_ENFORCE_EQ(
        map_value_pair->count(cur_in),
        true,
        common::errors::PreconditionNotMet(
            "[%d]'s input of [%s] op MUST in map pair", 0, op_item->name()));
    auto new_in = map_value_pair->at(cur_in);
    if (i == 0) {
      cond_val = new_in;
    } else {
      vec_in.push_back(new_in);
    }
  }

  pir::Builder builder(ctx, block);
  auto base_while_op = op_item->dyn_cast<WhileOp>();
  auto new_while_op = builder.Build<WhileOp>(cond_val, vec_in);
  pir::Block& body_block = new_while_op.body();
  for (size_t i = 0; i < vec_in.size(); ++i) {
    auto block_arg = body_block.arg(i);
    (*map_value_pair)[base_while_op.body().arg(i)] = block_arg;
  }

  // process body block
  ProcessBlock(place,
               &base_while_op.body(),
               &body_block,
               ctx,
               map_op_pair,
               map_value_pair);

  (*map_op_pair)[op_item] = new_while_op;

  // only deal with single output
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      (*map_value_pair)[op_item->result(i)] = new_while_op->result(i);
    }
  }
}

pir::Value GetNewInput(
    const pir::Value cur_in,
    const std::unordered_map<pir::Value, pir::Value>& map_value_pair,
    const int index,
    const std::string& op_name) {
  PADDLE_ENFORCE_EQ(
      map_value_pair.count(cur_in),
      true,
      common::errors::PreconditionNotMet(
          "[%d]'s input of [%s] op MUST be in map pair", index, op_name));
  auto new_in = map_value_pair.at(cur_in);
  return new_in;
}

phi::Place ParsePhiPlace(pir::Type type) {
  if (type.isa<AllocatedDenseTensorType>()) {
    return type.dyn_cast<AllocatedDenseTensorType>().place();
  } else if (type.isa<AllocatedSelectedRowsType>()) {
    return type.dyn_cast<AllocatedSelectedRowsType>().place();
  } else if (type.isa<AllocatedSparseCooTensorType>()) {
    return type.dyn_cast<AllocatedSparseCooTensorType>().place();
  } else if (type.isa<AllocatedSparseCsrTensorType>()) {
    return type.dyn_cast<AllocatedSparseCsrTensorType>().place();
  } else if (type.isa<AllocatedDenseTensorArrayType>()) {
    return type.dyn_cast<AllocatedDenseTensorArrayType>().place();
  } else if (type.isa<pir::VectorType>()) {
    return ParsePhiPlace(type.dyn_cast<pir::VectorType>()[0]);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "ParsePhiPlace only support AllocatedDenseTensorType or "
        "AllocatedSelectedRowsType or AllocatedSparseCooTensorType or "
        "AllocatedSparseCsrTensorType or AllocatedDenseTensorArrayType"));
  }
}

phi::DataType ParsePhiDType(pir::Type type) {
  if (type.isa<AllocatedDenseTensorType>()) {
    return TransToPhiDataType(
        type.dyn_cast<AllocatedDenseTensorType>().dtype());
  } else if (type.isa<AllocatedSelectedRowsType>()) {
    return TransToPhiDataType(
        type.dyn_cast<AllocatedSelectedRowsType>().dtype());
  } else if (type.isa<AllocatedSparseCooTensorType>()) {
    return TransToPhiDataType(
        type.dyn_cast<AllocatedSparseCooTensorType>().dtype());
  } else if (type.isa<AllocatedSparseCsrTensorType>()) {
    return TransToPhiDataType(
        type.dyn_cast<AllocatedSparseCsrTensorType>().dtype());
  } else if (type.isa<AllocatedDenseTensorArrayType>()) {
    return TransToPhiDataType(
        type.dyn_cast<AllocatedDenseTensorArrayType>().dtype());
  } else if (type.isa<pir::VectorType>()) {
    return ParsePhiDType(type.dyn_cast<pir::VectorType>()[0]);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "ParsePhiPlace only support AllocatedDenseTensorType or "
        "AllocatedSelectedRowsType or AllocatedDenseTensorArrayType or "
        "AllocatedSparseCooTensorType or AllocatedSparseCsrTensorType"));
  }
}

void AddShadowFeedForValue(
    size_t index,
    pir::Operation* op_item,
    pir::Operation* op_item_with_place,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  phi::Backend backend = paddle::experimental::get_accelerat_backend();

  if (op_item->result(index).type().isa<DenseTensorType>()) {
    phi::KernelKey shadow_key{
        backend,
        phi::DataLayout::ANY,
        TransToPhiDataType(
            op_item->result(index).type().dyn_cast<DenseTensorType>().dtype())};
    std::unordered_map<std::string, pir::Attribute> attr_map{
        {"op_name", pir::StrAttribute::get(ctx, "pd_op.shadow_feed")},
        {"kernel_name", pir::StrAttribute::get(ctx, "shadow_feed")},
        {"kernel_key", KernelAttribute::get(ctx, shadow_key)},
        {"dst_place_type", pir::Int32Attribute::get(ctx, 1)}};

    auto out_type = AllocatedDenseTensorType::get(
        ctx,
        phi::TransToPhiPlace(shadow_key.backend()),
        op_item->result(index).type().dyn_cast<DenseTensorType>());

    pir::OpInfo phi_kernel_op_info =
        ctx->GetRegisteredOpInfo(PhiKernelOp::name());
    pir::Operation* shadow_op =
        pir::Operation::Create({op_item_with_place->result(index)},
                               attr_map,
                               {out_type},
                               phi_kernel_op_info);
    shadow_op->set_attribute("origin_id",
                             pir::Int64Attribute::get(ctx, shadow_op->id()));

    block->push_back(shadow_op);
    (*map_op_pair)[op_item] = shadow_op;
    (*map_value_pair)[op_item->result(index)] = shadow_op->result(0);
  } else if (op_item->result(index).type().isa<pir::VectorType>()) {
    auto vec_type = op_item->result(index).type().dyn_cast<pir::VectorType>();
    for (size_t i = 0; i < vec_type.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          vec_type[i].isa<DenseTensorType>(),
          true,
          common::errors::PreconditionNotMet(
              "AddShadowFeedTensors only support DenseTensorType Now"));
    }
    // Add ShadowFeedTensors Op
    phi::KernelKey shadow_key{
        backend,
        phi::DataLayout::ANY,
        TransToPhiDataType(vec_type[0].dyn_cast<DenseTensorType>().dtype())};

    std::unordered_map<std::string, pir::Attribute> attr_map{
        {"op_name", pir::StrAttribute::get(ctx, "pd_op.shadow_feed_tensors")},
        {"kernel_name", pir::StrAttribute::get(ctx, "shadow_feed_tensors")},
        {"kernel_key", KernelAttribute::get(ctx, shadow_key)},
        {"dst_place_type", pir::Int32Attribute::get(ctx, 1)}};

    pir::OpInfo phi_kernel_op_info =
        ctx->GetRegisteredOpInfo(PhiKernelOp::name());

    std::vector<pir::Type> vec_out_types;
    for (size_t i = 0; i < vec_type.size(); ++i) {
      vec_out_types.push_back(AllocatedDenseTensorType::get(
          ctx,
          phi::TransToPhiPlace(shadow_key.backend()),
          vec_type[i].dyn_cast<DenseTensorType>()));
    }
    auto out_type = pir::VectorType::get(ctx, vec_out_types);
    pir::Operation* shadow_tensors_op =
        pir::Operation::Create({op_item_with_place->result(index)},
                               attr_map,
                               {out_type},
                               phi_kernel_op_info);
    shadow_tensors_op->set_attribute(
        "origin_id", pir::Int64Attribute::get(ctx, shadow_tensors_op->id()));
    block->push_back(shadow_tensors_op);
    (*map_op_pair)[op_item] = shadow_tensors_op;
    (*map_value_pair)[op_item->result(index)] = shadow_tensors_op->result(0);
  } else {
    PADDLE_THROW(
        common::errors::Unimplemented("AddShadowFeed for value only support "
                                      "DenseTensorType and VectorType Now"));
  }
}

void AddShadowFeedForTuplePopOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Operation* op_item_with_undefined_place,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  VLOG(4) << "Add AddShadowFeed for op " << op_item->name();

  bool add_shadow_feed = true;
  if (op_item->attributes().count("place")) {
    add_shadow_feed = (op_item->attributes()
                           .at("place")
                           .dyn_cast<PlaceAttribute>()
                           .data()
                           .GetType()) == phi::AllocationType::UNDEFINED;
  }

  // if value place not gpu, add shadow feed op
  if (phi::is_accelerat_place(place) && add_shadow_feed) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      AddShadowFeedForValue(i,
                            op_item,
                            op_item_with_undefined_place,
                            block,
                            ctx,
                            map_op_pair,
                            map_value_pair);
    }
  }
}

void HandleForSpecialOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair,
    bool for_if_block) {
  if (op_item->isa<IfOp>()) {
    HandleForIfOp(place, op_item, block, ctx, map_op_pair, map_value_pair);
    return;
  }

  if (op_item->isa<PyLayerOp>()) {
    HandleForPyLayerOp(place, op_item, block, ctx, map_op_pair, map_value_pair);
    return;
  }

  if (op_item->isa<WhileOp>()) {
    HandleForWhileOp(place, op_item, block, ctx, map_op_pair, map_value_pair);
    return;
  }

  std::vector<pir::Value> vec_inputs;
  std::vector<pir::Type> op_output_types;

  if (op_item->isa<::pir::CombineOp>()) {
    // Copy op inputs
    std::vector<pir::Type> vec_inner_types;
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
        vec_inputs.push_back(new_in);
        vec_inner_types.push_back(new_in.type());
      }
    }
    // Copy op output type

    pir::Type t1 = pir::VectorType::get(ctx, vec_inner_types);
    op_output_types.push_back(t1);
  }

  if (op_item->isa<::pir::ParameterOp>()) {
    op_output_types.push_back(
        BuildOutputType(op_item->result(0).type(), place, ctx));
  }

  if (op_item->isa<::pir::ConstantTensorOp>()) {
    op_output_types.push_back(
        BuildOutputType(op_item->result(0).type(), phi::CPUPlace(), ctx));
  }

  if (op_item->isa<::pir::SliceOp>()) {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
        vec_inputs.push_back(new_in);

        if (new_in.type().isa<pir::VectorType>()) {
          auto vec_types = new_in.type().dyn_cast<pir::VectorType>().data();
          auto index = op_item->attribute("index")
                           .dyn_cast<pir::Int32Attribute>()
                           .data();
          op_output_types.push_back(vec_types[index]);
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "only support vector type for now"));
        }
      }
    }
  }

  if (op_item->isa<::pir::SplitOp>()) {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
        vec_inputs.push_back(new_in);

        if (new_in.type().isa<pir::VectorType>()) {
          auto vec_types = new_in.type().dyn_cast<pir::VectorType>().data();
          for (uint64_t idx = 0; idx < vec_types.size(); idx++) {
            op_output_types.push_back(vec_types[idx]);
          }
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "only support vector type for now"));
        }
      }
    }
  }

  if (op_item->isa<::pir::YieldOp>()) {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());

        if (for_if_block && (!new_in.type().isa<pir::VectorType>()) &&
            (ParsePhiPlace(new_in.type()).GetType() !=
             phi::AllocationType::UNDEFINED) &&
            (ParsePhiPlace(new_in.type()) != place)) {
          phi::KernelKey kernel_key(TransToPhiBackend(place),
                                    phi::DataLayout::ALL_LAYOUT,
                                    ParsePhiDType(new_in.type()));
          new_in = AddPlaceTransferOp(
              new_in,
              ConvertOpTypeToKernelType(ctx, cur_in.type(), place),
              ParsePhiPlace(new_in.type()),
              place,
              kernel_key,
              block);
        }

        vec_inputs.push_back(new_in);
      }
    }
  }

  if (op_item->isa<::pir::ShadowOutputOp>()) {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());

        // layout transfer(only for onednn)
#ifdef PADDLE_WITH_DNNL
        auto new_in_type = new_in.type();
        if (new_in_type.isa<AllocatedDenseTensorType>()) {
          if (new_in_type.dyn_cast<AllocatedDenseTensorType>().data_layout() ==
              phi::DataLayout::ONEDNN) {
            new_in = AddOneDNN2PaddleLayoutTransferOp(
                new_in, phi::DataLayout::ANY, block);
          }
        }
#endif
        vec_inputs.push_back(new_in);
      }
    }
  }

  if (op_item->isa<::pir::SetParameterOp>()) {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
        // NOTE(zhangbo): parameter place is equal to exe place.
        if (new_in.type().isa<AllocatedDenseTensorType>()) {
          auto in_place =
              new_in.type().dyn_cast<AllocatedDenseTensorType>().place();
          auto dst_backend = phi::TransToPhiBackend(place);
          bool need_trans =
              (in_place.GetType() != phi::AllocationType::UNDEFINED) &&
              (paddle::experimental::NeedTransformPlace(
                  in_place, dst_backend, {}));
          if (need_trans) {
            VLOG(6) << "need trans from " << in_place << " to " << dst_backend;
            // build memcopy op
            auto out_place = phi::TransToPhiPlace(dst_backend);
            auto new_in_alloc_type =
                new_in.type().dyn_cast<AllocatedDenseTensorType>();
            auto out_type =
                AllocatedDenseTensorType::get(ctx,
                                              out_place,
                                              new_in_alloc_type.dtype(),
                                              new_in_alloc_type.dims(),
                                              new_in_alloc_type.data_layout(),
                                              new_in_alloc_type.lod(),
                                              new_in_alloc_type.offset());
            auto op_info_parser = GetOpYamlInfoParser(op_item);
            auto kernel_name = GetKernelName(op_info_parser.get(), op_item);
            auto kernel_key = GetKernelKey(op_item,
                                           place,
                                           kernel_name,
                                           *map_value_pair,
                                           op_info_parser.get());
            VLOG(6) << "kernel type " << kernel_key;
            new_in = AddPlaceTransferOp(
                new_in, out_type, in_place, out_place, kernel_key, block);
          }
        }
        vec_inputs.push_back(new_in);
      }
    }
  }

  if (op_item->isa<::pir::StackCreateOp>() ||
      op_item->isa<::pir::TuplePushOp>()) {
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      if (!cur_in) {
        vec_inputs.emplace_back();
        continue;
      }
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
      vec_inputs.push_back(new_in);
    }
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      op_output_types.push_back(op_item->result(i).type());
    }
  }

  if (op_item->isa<HasElementsOp>()) {
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
      vec_inputs.push_back(new_in);
    }
    PADDLE_ENFORCE_EQ(op_item->result(0).type().isa<DenseTensorType>(),
                      true,
                      common::errors::PreconditionNotMet(
                          "HasElementsOp's output should be bool type"));
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      op_output_types.push_back(
          BuildOutputType(op_item->result(i).type(), phi::CPUPlace(), ctx));
    }
  }

  if (op_item->isa<AssertOp>()) {
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
      vec_inputs.push_back(new_in);
    }
  }

  if (op_item->isa<::pir::TuplePopOp>()) {
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
      vec_inputs.push_back(new_in);
    }

    auto pop_back_op = op_item->dyn_cast<::pir::TuplePopOp>();

    if (pop_back_op.has_container()) {
      // if TuplePopOp and TuplePushOp are in the same sub_program
      for (size_t i = 0; i < op_item->num_results(); ++i) {
        auto cur_inlet_element = pop_back_op.inlet_element(i);
        PADDLE_ENFORCE_EQ(map_value_pair->count(cur_inlet_element),
                          true,
                          common::errors::PreconditionNotMet(
                              "[%d]'s output of [%s] op MUST be in map pair",
                              i,
                              op_item->name()));
        auto new_inlet_element = map_value_pair->at(cur_inlet_element);

        op_output_types.push_back(new_inlet_element.type());
      }
    } else {
      VLOG(4) << "TuplePopOp and TuplePushOp are in different sub_program.";
      for (size_t i = 0; i < op_item->num_results(); ++i) {
        auto cur_inlet_element = op_item->result(i);
        auto out_place = phi::TransToPhiPlace(phi::Backend::UNDEFINED);
        pir::Type new_inlet_element_type =
            ConvertOpTypeToKernelType(ctx, cur_inlet_element.type(), out_place);
        op_output_types.push_back(new_inlet_element_type);
      }

      pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_item->name());
      pir::Operation* op = pir::Operation::Create(
          vec_inputs, op_item->attributes(), op_output_types, op_info);
      op->set_attribute("origin_id", pir::Int64Attribute::get(ctx, op->id()));

      block->push_back(op);
      (*map_op_pair)[op_item] = op;
      // only deal with single output
      if (op_item->num_results() > 0) {
        for (size_t i = 0; i < op_item->num_results(); ++i) {
          (*map_value_pair)[op_item->result(i)] = op->result(i);
        }
      }
      AddShadowFeedForTuplePopOp(
          place, op_item, op, block, ctx, map_op_pair, map_value_pair);
      return;
    }
  }

  if (op_item->isa<SelectInputOp>()) {
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
      vec_inputs.push_back(new_in);
    }
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      op_output_types.push_back(vec_inputs[1].type());
    }
  }

  if (op_item->isa<SelectOutputOp>()) {
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
      vec_inputs.push_back(new_in);
    }
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      op_output_types.push_back(vec_inputs[1].type());
    }
  }

  if (op_item->name() == "cinn_runtime.jit_kernel") {
    std::vector<pir::Value> in_temps;
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      if (!cur_in) {
        in_temps.emplace_back();
        continue;
      }
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());

      in_temps.push_back(new_in);
    }

    auto dst_backend = phi::TransToPhiBackend(place);
    auto exec_backend = paddle::dialect::PlaceAttribute::get(ctx, place);

    bool run_cpu_kernel = CanGroupOpRunCpuKernel(in_temps, op_item->results());
#ifdef PADDLE_WITH_CINN

    cinn::dialect::JitKernelOp jit_kernel_op =
        op_item->dyn_cast<cinn::dialect::JitKernelOp>();
    const cinn::hlir::framework::pir::CINNKernelInfo& kernel_info =
        jit_kernel_op.cinn_kernel_info();
    if (run_cpu_kernel && kernel_info.CX86_fn_ptr == nullptr) {
      // change dst_backend to cpu
      run_cpu_kernel = false;
    }
#endif
    if (run_cpu_kernel) {
      dst_backend = phi::Backend::CPU;

      exec_backend = paddle::dialect::PlaceAttribute::get(
          ctx, phi::Place(phi::AllocationType::CPU));
    }

    op_item->set_attribute(kAttrExecBackend, exec_backend);

    for (size_t i = 0; i < in_temps.size(); ++i) {
      auto new_in = in_temps[i];
      // For data transform
      if (new_in.type().isa<AllocatedDenseTensorType>()) {
        auto in_place =
            new_in.type().dyn_cast<AllocatedDenseTensorType>().place();

        bool need_trans =
            (in_place.GetType() != phi::AllocationType::UNDEFINED) &&
            (paddle::experimental::NeedTransformPlace(
                in_place, dst_backend, {}));
        if (need_trans) {
          VLOG(6) << "need trans from " << in_place << " to " << dst_backend;
          auto value_type =
              op_item->operand_source(i).type().dyn_cast<DenseTensorType>();
          auto out_place = phi::TransToPhiPlace(dst_backend);
          auto out_type =
              AllocatedDenseTensorType::get(ctx, out_place, value_type);
          phi::KernelKey kernel_key(
              paddle::experimental::get_accelerat_backend(),
              phi::DataLayout::ANY,
              TransToPhiDataType(value_type.dtype()));
          new_in = AddPlaceTransferOp(
              new_in, out_type, in_place, out_place, kernel_key, block);
        }
      }
      vec_inputs.push_back(new_in);
    }

    for (size_t i = 0; i < op_item->num_results(); ++i) {
      op_output_types.push_back(AllocatedDenseTensorType::get(
          ctx,
          phi::TransToPhiPlace(dst_backend),
          op_item->result(i).type().dyn_cast<DenseTensorType>()));
    }
  }

  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_item->name());
  // Generate new op

  pir::Operation* op = pir::Operation::Create(
      vec_inputs, op_item->attributes(), op_output_types, op_info);
  op->set_attribute("origin_id", pir::Int64Attribute::get(ctx, op_item->id()));

  block->push_back(op);
  (*map_op_pair)[op_item] = op;
  // only deal with single output
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      (*map_value_pair)[op_item->result(i)] = op->result(i);
    }
  }
  VLOG(6) << "Deep copy a new special op: " << op_item->name();
}

void PushBackOutputTypes(pir::IrContext* ctx,
                         pir::Operation* op_item,
                         const pir::Type& origin_type,
                         const phi::Place& out_place,
                         const phi::KernelKey& kernel_key,
                         std::vector<pir::Type>* op_output_types) {
  auto result_type = origin_type;
  if (!result_type) {
    op_output_types->push_back(result_type);
  } else if (result_type.isa<DenseTensorType>() ||
             result_type.isa<SelectedRowsType>() ||
             result_type.isa<DenseTensorArrayType>() ||
             result_type.isa<SparseCooTensorType>() ||
             result_type.isa<SparseCsrTensorType>()) {
#ifdef PADDLE_WITH_DNNL
    if (kernel_key.backend() == phi::Backend::ONEDNN) {
      op_output_types->push_back(BuildOutputType(
          result_type, out_place, phi::DataLayout::ONEDNN, ctx));
    } else {
      op_output_types->push_back(BuildOutputType(result_type, out_place, ctx));
    }
#else
    op_output_types->push_back(BuildOutputType(result_type, out_place, ctx));
#endif

  } else if (result_type.isa<pir::VectorType>()) {
    std::vector<pir::Type> vec_inner_types;
    auto base_types = result_type.dyn_cast<pir::VectorType>().data();
    for (auto& base_type : base_types) {
      if (base_type) {
        if (base_type.isa<DenseTensorType>() ||
            base_type.isa<SelectedRowsType>()) {
#ifdef PADDLE_WITH_DNNL
          if (kernel_key.backend() == phi::Backend::ONEDNN) {
            vec_inner_types.push_back(BuildOutputType(
                base_type, out_place, phi::DataLayout::ONEDNN, ctx));
          } else {
            vec_inner_types.push_back(
                BuildOutputType(base_type, out_place, ctx));
          }
#else
          vec_inner_types.push_back(BuildOutputType(base_type, out_place, ctx));
#endif
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "only support dense tensor and selected rows in vector type "
              "for now"));
        }
      } else {
        // NOTE(phlrain), kernel not support a nullptr in output
        pir::Type fp32_dtype = pir::Float32Type::get(ctx);
        phi::DDim dims = {};
        phi::DataLayout data_layout = phi::DataLayout::NCHW;
#ifdef PADDLE_WITH_DNNL
        if (kernel_key.backend() == phi::Backend::ONEDNN) {
          data_layout = phi::DataLayout::ONEDNN;
        }
#endif
        phi::LegacyLoD lod = {{}};
        size_t offset = 0;
        auto dense_tensor_dtype = DenseTensorType::get(
            ctx, fp32_dtype, dims, data_layout, lod, offset);
        auto allocated_dense_tensor_dtype =
            AllocatedDenseTensorType::get(ctx, out_place, dense_tensor_dtype);
        vec_inner_types.push_back(allocated_dense_tensor_dtype);
      }
    }

    pir::Type t1 = pir::VectorType::get(ctx, vec_inner_types);
    op_output_types->push_back(t1);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Result type only support DenseTensorType, SelectedRowType, "
        "SparseCooTensorType, SparseCsrTensorType and "
        "VectorType"));
  }
}

void HandleForCustomOp(
    pir::IrContext* ctx,
    pir::Operation* op_item,
    const phi::KernelKey& kernel_key,
    const phi::Place place,
    const OpYamlInfoParser* op_info_parser,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair,
    pir::Block* block) {
  // Prepare output types
  std::vector<pir::Type> op_output_types;

  for (size_t i = 0; i < op_item->num_results(); ++i) {
    phi::Place out_place = phi::TransToPhiPlace(kernel_key.backend());
    PushBackOutputTypes(ctx,
                        op_item,
                        op_item->result(i).type(),
                        out_place,
                        kernel_key,
                        &op_output_types);
  }

  // Prepare input
  std::vector<pir::Value> vec_inputs;

  for (size_t i = 0; i < op_item->num_operands(); ++i) {
    auto cur_in = op_item->operand_source(i);
    if (!cur_in) {
      vec_inputs.emplace_back();
      continue;
    }
    PADDLE_ENFORCE_EQ(
        map_value_pair->count(cur_in),
        true,
        common::errors::PreconditionNotMet(
            "[%d]'s input of [%s] op MUST in map pair", i, op_item->name()));

    auto new_in = map_value_pair->at(cur_in);
    auto new_in_type = new_in.type();

    if (new_in_type.isa<AllocatedDenseTensorType>()) {
      auto in_place = new_in_type.dyn_cast<AllocatedDenseTensorType>().place();
      // GPU_PINNED -> GPU, refer to PR#41972
      if (phi::AllocationType::GPUPINNED == place.GetType()) {
        VLOG(6) << "need trans from GPUPINNED to GPU";
        // build memcopy op
        auto out_place = phi::TransToPhiPlace(phi::Backend::GPU);
        auto new_in_alloc_type =
            new_in_type.dyn_cast<AllocatedDenseTensorType>();
        auto out_type =
            AllocatedDenseTensorType::get(ctx,
                                          out_place,
                                          new_in_alloc_type.dtype(),
                                          new_in_alloc_type.dims(),
                                          new_in_alloc_type.data_layout(),
                                          new_in_alloc_type.lod(),
                                          new_in_alloc_type.offset());
        new_in = AddPlaceTransferOp(
            new_in, out_type, in_place, out_place, kernel_key, block);
      }
    }

    vec_inputs.push_back(new_in);
  }

  // Prepare attr
  std::unordered_map<std::string, pir::Attribute> op_attribute{
      {"op_name", pir::StrAttribute::get(ctx, op_item->name())},
      {"kernel_name", pir::StrAttribute::get(ctx, op_item->name())},
      {"kernel_key", KernelAttribute::get(ctx, kernel_key)}};
  auto op_attr_map = op_item->attributes();

  for (auto& map_item : op_attr_map) {
    op_attribute.emplace(map_item.first, map_item.second);
  }

  if (op_item->HasTrait<InplaceTrait>()) {
    op_attribute.emplace("is_inplace", pir::BoolAttribute::get(ctx, true));
  }

  op_attribute.emplace("origin_id",
                       pir::Int64Attribute::get(ctx, op_item->id()));

  VLOG(6) << "Lower custom op: " << op_item->name()
          << " to : " << CustomKernelOp::name();

  pir::OpInfo custom_kernel_op_info =
      ctx->GetRegisteredOpInfo(CustomKernelOp::name());

  pir::Operation* op = nullptr;
  op = pir::Operation::Create(
      vec_inputs, op_attribute, op_output_types, custom_kernel_op_info);
  op->set_attribute("origin_id", pir::Int64Attribute::get(ctx, op->id()));

  (*map_op_pair)[op_item] = op;

  // only deal with single output
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      (*map_value_pair)[op_item->result(i)] = op->result(i);
    }
  }
  block->push_back(op);
}

void HandleForTensorRTOp(
    pir::IrContext* ctx,
    pir::Operation* op_item,
    const phi::KernelKey& kernel_key,
    const phi::Place place,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair,
    pir::Block* block) {
  // Prepare output types
  std::vector<pir::Type> op_output_types;

  for (size_t i = 0; i < op_item->num_results(); ++i) {
    phi::Place out_place = phi::TransToPhiPlace(kernel_key.backend());
    PushBackOutputTypes(ctx,
                        op_item,
                        op_item->result(i).type(),
                        out_place,
                        kernel_key,
                        &op_output_types);
  }

  // Prepare input
  std::vector<pir::Value> vec_inputs;

  for (size_t i = 0; i < op_item->num_operands(); ++i) {
    auto cur_in = op_item->operand_source(i);
    PADDLE_ENFORCE_EQ(
        map_value_pair->count(cur_in),
        true,
        common::errors::PreconditionNotMet(
            "[%d]'s input of [%s] op MUST in map pair", i, op_item->name()));

    auto new_in = map_value_pair->at(cur_in);

    vec_inputs.push_back(new_in);
  }

  // Prepare attr
  std::unordered_map<std::string, pir::Attribute> op_attribute;
  auto op_attr_map = op_item->attributes();
  for (auto& map_item : op_attr_map) {
    op_attribute.emplace(map_item.first, map_item.second);
  }
  op_attribute["op_name"] = pir::StrAttribute::get(ctx, op_item->name());

  pir::OpInfo trt_op_info = ctx->GetRegisteredOpInfo(TensorRTEngineOp::name());

  pir::Operation* op = nullptr;
  op = pir::Operation::Create(
      vec_inputs, op_attribute, op_output_types, trt_op_info);
  op->set_attribute("origin_id", pir::Int64Attribute::get(ctx, op->id()));

  (*map_op_pair)[op_item] = op;

  // only deal with single output
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      (*map_value_pair)[op_item->result(i)] = op->result(i);
    }
  }
  block->push_back(op);
}

#ifdef PADDLE_WITH_CUSTOM_DEVICE
void HandleForCustomEngineOp(
    pir::IrContext* ctx,
    pir::Operation* op_item,
    phi::KernelKey* kernel_key,
    phi::Place place,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair,
    pir::Block* block,
    C_CustomEngineInterface* interface) {
  if (interface->custom_engine_op_lower) {
    struct C_CustomEngineLowerParams lower_params {
      reinterpret_cast<C_IrContext>(ctx),
          reinterpret_cast<C_Operation>(op_item),
          reinterpret_cast<C_KernelKey>(kernel_key),
          reinterpret_cast<C_Place>(&place),
          reinterpret_cast<C_Operation_Map>(map_op_pair),
          reinterpret_cast<C_Value_Map>(map_value_pair),
          reinterpret_cast<C_Block>(block)
    };
    VLOG(6) << "Handle CustomEngineOp while lowering to kernel pass";
    interface->custom_engine_op_lower(&lower_params);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "CustomEngineInstruction's "
        "C_CustomEngineInterface->custom_engine_op_lower not implemented"));
  }
}
#endif
std::vector<pir::Type> BuildOutputs(
    pir::Operation* op_item,
    const std::string& kernel_fn_str,
    const phi::KernelKey& kernel_key,
    const std::vector<pir::Value>& new_vec_inputs,
    pir::IrContext* ctx) {
  if (op_item->num_results() == 0) {
    return {};
  }
  std::vector<pir::Type> op_output_types;
  pir::AttributeMap attribute_map = op_item->attributes();

  auto phi_kernel =
      phi::KernelFactory::Instance().SelectKernel(kernel_fn_str, kernel_key);
  VLOG(6) << "[" << kernel_fn_str
          << "] selected kernel(is_valid: " << phi_kernel.IsValid()
          << " ): " << kernel_key;

  auto args_def = phi_kernel.args_def();
  auto output_defs = args_def.output_defs();

  if (!UnchangeOutputOps.count(op_item->name()) &&
      !IsLegacyOp(op_item->name())) {
    PADDLE_ENFORCE_EQ(
        op_item->num_results(),
        output_defs.size(),
        common::errors::PreconditionNotMet(
            "op [%s] kernel output args (%d) defs should equal op outputs (%d)",
            op_item->name(),
            output_defs.size(),
            op_item->num_results()));
  }

  bool is_input_type_changed = false;
  for (size_t i = 0; i < op_item->num_operands(); ++i) {
    if (GetValueDataType(op_item->operand(i).source()) !=
        GetValueDataType(new_vec_inputs[i])) {
      is_input_type_changed = true;
      break;
    }
  }

  bool is_custom_set = false;
  if (is_input_type_changed) {
    std::vector<pir::Value> input_values;
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      input_values.emplace_back(op_item->operand(i).source());
    }
    std::vector<pir::Type> output_types =
        InferMetaByValue(op_item, input_values, &attribute_map);

    if (output_types.size() != 0) {
      PADDLE_ENFORCE_EQ(
          output_types.size(),
          op_item->num_results(),
          common::errors::PreconditionNotMet(
              "output_types.size() is expected to be %d but got %d",
              op_item->num_results(),
              output_types.size()));
      for (size_t i = 0; i < op_item->num_results(); ++i) {
        if (output_types[i] != op_item->result(i).type()) {
          is_custom_set = true;
          break;
        }
      }
    }
  }

  if (!is_input_type_changed || is_custom_set) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      phi::Place out_place = phi::TransToPhiPlace(kernel_key.backend());
      if ((!UnchangeOutputOps.count(op_item->name())) &&
          (!IsLegacyOp(op_item->name())) && phi_kernel.IsValid()) {
        out_place = phi::TransToPhiPlace(output_defs[i].backend);
      }
      if (op_item->isa<MemcpyOp>()) {
        // If the op is MemcpyOp, the output type is determined by the
        // attribute "dst_place_type".
        out_place = MemcpyOpAttr2Place.at(op_item->attribute("dst_place_type")
                                              .dyn_cast<pir::Int32Attribute>()
                                              .data());
      }
      PushBackOutputTypes(ctx,
                          op_item,
                          op_item->result(i).type(),
                          out_place,
                          kernel_key,
                          &op_output_types);
    }
  } else {
    auto base_types = InferMetaByValue(op_item, new_vec_inputs, &attribute_map);
    PADDLE_ENFORCE_EQ(base_types.size(),
                      op_item->num_results(),
                      common::errors::PreconditionNotMet(
                          "base_types.size() is expected to be %d but got %d",
                          op_item->num_results(),
                          base_types.size()));
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      phi::Place out_place = phi::TransToPhiPlace(kernel_key.backend());
      if ((!UnchangeOutputOps.count(op_item->name())) &&
          (!IsLegacyOp(op_item->name())) && phi_kernel.IsValid()) {
        out_place = phi::TransToPhiPlace(output_defs[i].backend);
      }
      PushBackOutputTypes(
          ctx, op_item, base_types[i], out_place, kernel_key, &op_output_types);
    }
  }
  return op_output_types;
}

std::vector<pir::Value> BuildInputs(
    pir::Operation* op_item,
    const std::string& kernel_fn_str,
    const phi::KernelKey& kernel_key,
    const phi::Place place,
    const OpYamlInfoParser* op_info_parser,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair,
    pir::Block* block) {
  if (op_item->num_operands() == 0) {
    return {};
  }

  std::vector<pir::Value> vec_inputs;

  for (size_t i = 0; i < op_item->num_operands(); ++i) {
    auto cur_in = op_item->operand_source(i);
    if (!cur_in) {
      vec_inputs.emplace_back();
      continue;
    }
    PADDLE_ENFORCE_EQ(
        map_value_pair->count(cur_in),
        true,
        common::errors::PreconditionNotMet(
            "[%d]'s input of [%s] op MUST in map pair", i, op_item->name()));

    auto new_in = map_value_pair->at(cur_in);
    auto new_in_type = new_in.type();

    auto& kernel = phi::KernelFactory::Instance().SelectKernelWithGPUDNN(
        kernel_fn_str, kernel_key);

    int tensor_param_index = static_cast<int>(i);
    if (kernel.IsValid()) {
      tensor_param_index = op_info_parser->GetTensorParamIndexByArgsName(
          op_info_parser->InputNames()[i]);
      // the input of op args is not the kernel parameter
      if (tensor_param_index == -1) {
        vec_inputs.emplace_back(new_in);
        continue;
      }
    }

    // 1. layout transfer(only for onednn)
#ifdef PADDLE_WITH_DNNL
    if (kernel_key.backend() != phi::Backend::ONEDNN) {
      auto new_in_type = new_in.type();
      if (new_in_type.isa<AllocatedDenseTensorType>()) {
        if (new_in_type.dyn_cast<AllocatedDenseTensorType>().data_layout() ==
            phi::DataLayout::ONEDNN) {
          new_in = AddOneDNN2PaddleLayoutTransferOp(
              new_in, phi::DataLayout::ANY, block);
        }
      } else if (new_in_type.isa<pir::VectorType>() &&
                 new_in.defining_op()->isa<::pir::CombineOp>()) {
        bool need_replace_combine_op = false;
        std::vector<pir::Value> new_vec_inputs;
        std::vector<pir::Type> types_in_vec;
        for (auto& in : new_in.defining_op()->operands()) {
          auto in_value = in.source();
          if (in_value.type().isa<AllocatedDenseTensorType>()) {
            if (in_value.type()
                    .dyn_cast<AllocatedDenseTensorType>()
                    .data_layout() == phi::DataLayout::ONEDNN) {
              need_replace_combine_op = true;
              in_value = AddOneDNN2PaddleLayoutTransferOp(
                  in_value, phi::DataLayout::ANY, block);
            }
            new_vec_inputs.push_back(in_value);
            types_in_vec.push_back(in_value.type());
          }
        }
        if (need_replace_combine_op) {
          std::string combine_op_name(pir::CombineOp::name());
          pir::OpInfo op_info = ctx->GetRegisteredOpInfo(combine_op_name);

          pir::Type target_vec_type = pir::VectorType::get(ctx, types_in_vec);
          pir::Operation* operation = pir::Operation::Create(
              new_vec_inputs, {}, {target_vec_type}, op_info);
          operation->set_attribute(
              "origin_id", pir::Int64Attribute::get(ctx, operation->id()));
          new_in.defining_op()->ReplaceAllUsesWith(operation->results());
          block->erase(*new_in.defining_op());

          new_in = operation->result(0);
          block->push_back(operation);
        }
      }
    }
#endif

    // 2.backend transfer
    bool check_place_transfer =
        (op_item->isa<::pir::SetParameterOp>()) ||
        (kernel.IsValid() && (!UnchangeOutputOps.count(op_item->name())));

    // NOTE(Aurelius84): In case of Reshape/Squeeze/Flatten.XShape,
    // Skip insert memcpyd2h for fetch op
    check_place_transfer =
        check_place_transfer && !NeedSkipPlaceTransfer(op_item);

    if (check_place_transfer) {
      if (new_in_type.isa<AllocatedDenseTensorType>()) {
        // allocated type
        auto in_place =
            new_in_type.dyn_cast<AllocatedDenseTensorType>().place();

        // get input args def type
        auto args_def = kernel.args_def();
        auto input_defs = args_def.input_defs();

        auto input_backend = ChooseInputBackend(
            kernel, tensor_param_index, kernel_key.backend());
        auto dst_backend = DeriveBackend(
            op_item->name(), place, op_info_parser, input_backend, i);
        VLOG(6) << "Infer kernel backend from input " << i << " of op "
                << op_item->name();

        bool need_trans =
            (in_place.GetType() != phi::AllocationType::UNDEFINED) &&
            (paddle::experimental::NeedTransformPlace(
                in_place, dst_backend, {}));
        if (need_trans) {
          VLOG(6) << "need trans from " << in_place << " to "
                  << kernel_key.backend();
          // build memcopy op
          auto out_place = phi::TransToPhiPlace(dst_backend);
          auto new_in_alloc_type =
              new_in_type.dyn_cast<AllocatedDenseTensorType>();
          auto out_type =
              AllocatedDenseTensorType::get(ctx,
                                            out_place,
                                            new_in_alloc_type.dtype(),
                                            new_in_alloc_type.dims(),
                                            new_in_alloc_type.data_layout(),
                                            new_in_alloc_type.lod(),
                                            new_in_alloc_type.offset());
          new_in = AddPlaceTransferOp(
              new_in, out_type, in_place, out_place, kernel_key, block);
        }
      } else if (new_in_type.isa<pir::VectorType>()) {
        // [ todo need update here, support combine data transformer]
        // deal with pre combine op
        auto pre_define_op = cur_in.defining_op();
        if (pre_define_op->isa<::pir::CombineOp>()) {
          std::vector<pir::Value> inner_inputs;
          std::vector<pir::Type> types_in_vec;
          bool is_trans = false;
          for (size_t j = 0; j < pre_define_op->num_operands(); ++j) {
            auto in_i = map_value_pair->at(pre_define_op->operand_source(j));
            auto in_i_type = in_i.type();
            phi::Place place;
            if (in_i_type.isa<AllocatedDenseTensorType>()) {
              place = in_i_type.dyn_cast<AllocatedDenseTensorType>().place();
            } else if (in_i_type.isa<AllocatedSelectedRowsType>()) {
              place = in_i_type.dyn_cast<AllocatedSelectedRowsType>().place();
            } else if (in_i_type.isa<AllocatedSparseCooTensorType>()) {
              place =
                  in_i_type.dyn_cast<AllocatedSparseCooTensorType>().place();
            } else if (in_i_type.isa<AllocatedSparseCsrTensorType>()) {
              place =
                  in_i_type.dyn_cast<AllocatedSparseCsrTensorType>().place();
            } else if (in_i_type.isa<AllocatedDenseTensorArrayType>()) {
              place =
                  in_i_type.dyn_cast<AllocatedDenseTensorArrayType>().place();
            } else {
              PADDLE_THROW(common::errors::Unimplemented(
                  "builtin.combine Input type only support "
                  "VectorType<DenseTensorType> and "
                  "VectorType<SelectedRowsType> and"
                  "VectorType<DenseTensorArrayType> and"
                  "VectorType<SparseCooTensorType> and"
                  "VectorType<SparseCsrTensorType>"));
            }

            // get input args def type
            auto args_def = kernel.args_def();
            auto input_defs = args_def.input_defs();

            auto input_backend = ChooseInputBackend(
                kernel, tensor_param_index, kernel_key.backend());
            bool need_trans =
                (place.GetType() != phi::AllocationType::UNDEFINED) &&
                (op_info_parser != nullptr &&
                 !op_info_parser->IsTensorAttribute(i)) &&
                (paddle::experimental::NeedTransformPlace(
                    place, input_backend, {}));
            if (need_trans) {
              // build memcopy op
              auto out_place = phi::TransToPhiPlace(input_backend);
              pir::Type out_type;
              if (in_i_type.isa<AllocatedDenseTensorType>()) {
                out_type = AllocatedDenseTensorType::get(
                    ctx,
                    out_place,
                    pre_define_op->operand_source(j)
                        .type()
                        .dyn_cast<DenseTensorType>());
              } else if (in_i_type.isa<AllocatedSelectedRowsType>()) {
                out_type = AllocatedSelectedRowsType::get(
                    ctx,
                    out_place,
                    pre_define_op->operand_source(j)
                        .type()
                        .dyn_cast<SelectedRowsType>());
              } else if (in_i_type.isa<AllocatedSparseCooTensorType>()) {
                out_type = AllocatedSparseCooTensorType::get(
                    ctx,
                    out_place,
                    pre_define_op->operand_source(j)
                        .type()
                        .dyn_cast<SparseCooTensorType>());
              } else if (in_i_type.isa<AllocatedSparseCsrTensorType>()) {
                out_type = AllocatedSparseCsrTensorType::get(
                    ctx,
                    out_place,
                    pre_define_op->operand_source(j)
                        .type()
                        .dyn_cast<SparseCsrTensorType>());
              } else if (in_i_type.isa<AllocatedDenseTensorArrayType>()) {
                out_type = AllocatedDenseTensorArrayType::get(
                    ctx,
                    out_place,
                    pre_define_op->operand_source(j)
                        .type()
                        .dyn_cast<DenseTensorArrayType>());
              } else {
                PADDLE_THROW(common::errors::Unimplemented(
                    "builtin.combine Input type only support "
                    "VectorType<DenseTensorType> and "
                    "VectorType<SelectedRowsType> and"
                    "VectorType<DenseTensorArrayType> and"
                    "VectorType<SparseCooTensorType> and"
                    "VectorType<SparseCsrTensorType>"));
              }
              in_i = AddPlaceTransferOp(
                  in_i, out_type, place, out_place, kernel_key, block);

              is_trans = true;
            }

            inner_inputs.push_back(in_i);
            types_in_vec.push_back(in_i.type());
          }
          if (is_trans) {
            // Add combine op
            std::string combine_op_name(pir::CombineOp::name());
            pir::OpInfo op_info = ctx->GetRegisteredOpInfo(combine_op_name);

            pir::Type target_vec_type = pir::VectorType::get(ctx, types_in_vec);
            pir::Operation* operation = pir::Operation::Create(
                inner_inputs, {}, {target_vec_type}, op_info);
            operation->set_attribute(
                "origin_id", pir::Int64Attribute::get(ctx, operation->id()));

            new_in = operation->result(0);
            block->push_back(operation);
          }
        }
      } else if (new_in_type.isa<AllocatedSelectedRowsType>()) {
        // allocated type
        auto in_place =
            new_in_type.dyn_cast<AllocatedSelectedRowsType>().place();

        // get input args def type
        auto args_def = kernel.args_def();
        auto input_defs = args_def.input_defs();

        auto input_backend = ChooseInputBackend(
            kernel, tensor_param_index, kernel_key.backend());
        auto dst_backend = DeriveBackend(
            op_item->name(), place, op_info_parser, input_backend, i);
        VLOG(6) << "Infer kernel backend from input " << i << " of op ";
        bool need_trans =
            (in_place.GetType() != phi::AllocationType::UNDEFINED) &&
            (paddle::experimental::NeedTransformPlace(
                in_place, dst_backend, {}));
        if (need_trans) {
          VLOG(6) << "need trans from " << in_place << " to "
                  << kernel_key.backend();
          // build memcopy op
          auto out_place = phi::TransToPhiPlace(dst_backend);
          auto new_in_alloc_type =
              new_in_type.dyn_cast<AllocatedSelectedRowsType>();
          auto out_type =
              AllocatedSelectedRowsType::get(ctx,
                                             out_place,
                                             new_in_alloc_type.dtype(),
                                             new_in_alloc_type.dims(),
                                             new_in_alloc_type.data_layout(),
                                             new_in_alloc_type.lod(),
                                             new_in_alloc_type.offset());
          new_in = AddPlaceTransferOp(
              new_in, out_type, in_place, out_place, kernel_key, block);
        }
      } else if (new_in_type.isa<AllocatedDenseTensorArrayType>()) {
        // allocated type
        auto in_place =
            new_in_type.dyn_cast<AllocatedDenseTensorArrayType>().place();

        // get input args def type
        auto args_def = kernel.args_def();
        auto input_defs = args_def.input_defs();

        auto input_backend = ChooseInputBackend(
            kernel, tensor_param_index, kernel_key.backend());
        auto dst_backend = DeriveBackend(
            op_item->name(), place, op_info_parser, input_backend, i);
        VLOG(6) << "Infer kernel backend from input " << i << " of op ";
        bool need_trans =
            (in_place.GetType() != phi::AllocationType::UNDEFINED) &&
            (paddle::experimental::NeedTransformPlace(
                in_place, dst_backend, {}));
        if (need_trans) {
          VLOG(6) << "need trans from " << in_place << " to "
                  << kernel_key.backend();
          // build memcopy op
          auto out_place = phi::TransToPhiPlace(dst_backend);
          auto new_in_alloc_type =
              new_in_type.dyn_cast<AllocatedDenseTensorArrayType>();
          auto out_type = AllocatedDenseTensorArrayType::get(
              ctx,
              out_place,
              new_in_alloc_type.dtype(),
              new_in_alloc_type.dims(),
              new_in_alloc_type.data_layout());
          new_in = AddPlaceTransferOp(
              new_in, out_type, in_place, out_place, kernel_key, block);
        }
      } else if (new_in_type.isa<AllocatedSparseCooTensorType>()) {
        // allocated type
        auto in_place =
            new_in_type.dyn_cast<AllocatedSparseCooTensorType>().place();

        // get input args def type
        auto args_def = kernel.args_def();
        auto input_defs = args_def.input_defs();

        auto input_backend = ChooseInputBackend(
            kernel, tensor_param_index, kernel_key.backend());
        auto dst_backend = DeriveBackend(
            op_item->name(), place, op_info_parser, input_backend, i);
        VLOG(6) << "Infer kernel backend from input " << i << " of op ";
        bool need_trans =
            (in_place.GetType() != phi::AllocationType::UNDEFINED) &&
            (paddle::experimental::NeedTransformPlace(
                in_place, dst_backend, {}));
        if (need_trans) {
          VLOG(6) << "need trans from " << in_place << " to "
                  << kernel_key.backend();
          // build memcopy op
          auto out_place = phi::TransToPhiPlace(dst_backend);
          auto new_in_alloc_type =
              new_in_type.dyn_cast<AllocatedSparseCooTensorType>();
          auto out_type = AllocatedSparseCooTensorType::get(
              ctx,
              out_place,
              new_in_alloc_type.dtype(),
              new_in_alloc_type.dims(),
              new_in_alloc_type.non_zero_dims(),
              new_in_alloc_type.data_layout(),
              new_in_alloc_type.non_zero_indices(),
              new_in_alloc_type.non_zero_elements(),
              new_in_alloc_type.coalesced());
          new_in = AddPlaceTransferOp(
              new_in, out_type, in_place, out_place, kernel_key, block);
        }
      } else if (new_in_type.isa<AllocatedSparseCsrTensorType>()) {
        // allocated type
        auto in_place =
            new_in_type.dyn_cast<AllocatedSparseCsrTensorType>().place();

        // get input args def type
        auto args_def = kernel.args_def();
        auto input_defs = args_def.input_defs();

        auto input_backend = ChooseInputBackend(
            kernel, tensor_param_index, kernel_key.backend());
        auto dst_backend = DeriveBackend(
            op_item->name(), place, op_info_parser, input_backend, i);
        VLOG(6) << "Infer kernel backend from input " << i << " of op ";
        bool need_trans =
            (in_place.GetType() != phi::AllocationType::UNDEFINED) &&
            (paddle::experimental::NeedTransformPlace(
                in_place, dst_backend, {}));
        if (need_trans) {
          VLOG(6) << "need trans from " << in_place << " to "
                  << kernel_key.backend();
          // build memcopy op
          auto out_place = phi::TransToPhiPlace(dst_backend);
          auto new_in_alloc_type =
              new_in_type.dyn_cast<AllocatedSparseCsrTensorType>();
          auto out_type = AllocatedSparseCsrTensorType::get(
              ctx,
              out_place,
              new_in_alloc_type.dtype(),
              new_in_alloc_type.dims(),
              new_in_alloc_type.data_layout(),
              new_in_alloc_type.non_zero_crows(),
              new_in_alloc_type.non_zero_cols(),
              new_in_alloc_type.non_zero_elements());
          new_in = AddPlaceTransferOp(
              new_in, out_type, in_place, out_place, kernel_key, block);
        }
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "only support AllocatedDenseTensorType, VectorType, "
            "AllocatedSelectedRowsType, AllocatedDenseTensorArrayType, "
            "AllocatedSparseCooTensorType and AllocatedSparseCsrTensorType"));
      }
    }

    // 3. dtype transfer
    if (op_info_parser != nullptr) {
      std::string var_name = op_info_parser->InputNames()[i];
      auto fake_tensors = PrepareFakeTensors(new_in);
      if (!fake_tensors.empty()) {
        const phi::KernelKey expected_kernel_key = kernel_key;
        const phi::DataType kernel_dtype_for_var =
            GetKernelTypeforVar(op_item,
                                var_name,
                                (*fake_tensors[0]).dtype(),
                                &expected_kernel_key);

        bool check_dtype_transfer = NeedTransformDataType(
            expected_kernel_key.dtype(), kernel_dtype_for_var);
        if (check_dtype_transfer) {
          VLOG(4) << "trans input: " << var_name << "'s dtype from "
                  << kernel_dtype_for_var << " to "
                  << expected_kernel_key.dtype();

          auto out_place = phi::TransToPhiPlace(expected_kernel_key.backend());
          new_in = AddDtypeTransferOp(new_in,
                                      block,
                                      kernel_key,
                                      place,
                                      out_place,
                                      kernel_dtype_for_var,
                                      expected_kernel_key.dtype());
        }
      }
    }

    vec_inputs.push_back(new_in);
  }
  return vec_inputs;
}

void AddShadowFeedOpForDataOrFeed(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Operation* kernel_op,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  bool feed_op_add_shadow_feed =
      (op_item->isa<FeedOp>()) && phi::is_accelerat_place(place);
  bool data_op_add_shadow_feed =
      (op_item->isa<DataOp>()) && phi::is_accelerat_place(place) &&
      (kernel_op->attributes()
           .at("place")
           .dyn_cast<PlaceAttribute>()
           .data()
           .GetType() == phi::AllocationType::UNDEFINED);
  bool add_shadow_feed = feed_op_add_shadow_feed || data_op_add_shadow_feed;
  if (add_shadow_feed) {
    PADDLE_ENFORCE(op_item->num_results() == 1,
                   common::errors::PreconditionNotMet(
                       "op_item should have only one result, but got %d",
                       op_item->num_results()));
    AddShadowFeedForValue(
        0, op_item, kernel_op, block, ctx, map_op_pair, map_value_pair);
  }
}

/*
   shadow_feed(x), y = memcpy_d2h(x), any_op(y)
=> shadow_feed(x, dst_place=cpu_place), any_op(x)

   shadow_feed(x), y = memcpy_h2d(x), any_op(y)
=> shadow_feed(x, dst_place=gpu_place), any_op(x)
*/
void RemoveRedundantMemcpyAfterShadowFeed(pir::Block* block,
                                          pir::IrContext* ctx) {
  for (auto it = block->begin(); it != block->end(); ++it) {
    if (it->isa<PhiKernelOp>() &&
        (it->dyn_cast<PhiKernelOp>().op_name() == "pd_op.shadow_feed")) {
      pir::Value shadow_value = it->result(0);
      if (shadow_value.use_count() == 1) {
        pir::Operation* next_op = shadow_value.first_use().owner();
        bool is_memcpy_d2h =
            next_op->isa<PhiKernelOp>() &&
            next_op->dyn_cast<PhiKernelOp>().op_name() == "pd_op.memcpy_d2h";
        bool is_memcpy_h2d =
            next_op->isa<PhiKernelOp>() &&
            next_op->dyn_cast<PhiKernelOp>().op_name() == "pd_op.memcpy_h2d";

        if (is_memcpy_d2h || is_memcpy_h2d) {
          VLOG(6) << "Remove redundant memcpy op after shadow_feed";
          VLOG(6) << *it;
          VLOG(6) << next_op;
          VLOG(6) << "==>";

          // remove memcpy op
          next_op->result(0).ReplaceAllUsesWith(shadow_value);
          block->erase(next_op->operator pir::Block::ConstIterator());

          // set dst_place_type for shadow_feed, 0 for cpu_place, 1 for
          // gpu_place
          int dst_place_type = is_memcpy_d2h ? 0 : 1;
          it->set_attribute("dst_place_type",
                            pir::Int32Attribute::get(ctx, dst_place_type));
          VLOG(6) << *it;
        }
      }

      if (shadow_value.use_count() >= 1) {
        bool all_use_is_scalar = true;
        for (auto use_it = shadow_value.use_begin();
             use_it != shadow_value.use_end();
             ++use_it) {
          auto use_op = use_it->owner();

          if (!use_op->isa<PhiKernelOp>()) {
            all_use_is_scalar = false;
            break;
          }
          auto op_name = use_op->dyn_cast<PhiKernelOp>().op_name();
          if (op_name == "pd_op.share_var") {
            all_use_is_scalar = false;
            break;
          }
          auto op_info = ctx->GetRegisteredOpInfo(op_name);

          if (!op_info) {
            all_use_is_scalar = false;
            break;
          }

          auto* op_info_concept =
              op_info.GetInterfaceImpl<dialect::OpYamlInfoInterface>();
          auto [input_infos, _1, _2, _3, _4] =
              op_info_concept->get_op_info_(op_info.name());

          uint32_t val_index = 0;
          for (uint32_t index = 0; index < use_op->num_operands(); index++) {
            if (use_op->operand_source(index) == shadow_value) {
              val_index = index;
              break;
            }
          }

          if (!input_infos[val_index].is_mutable_attribute) {
            all_use_is_scalar = false;
            break;
          }
        }
        if (all_use_is_scalar) {
          // set dst_place_type for shadow_feed, 0 for cpu_place
          VLOG(6) << "Reset shadow_feed dst_place_type to 0 for scalar use";
          it->set_attribute("dst_place_type", pir::Int32Attribute::get(ctx, 0));
        }
      }
    }
  }
}

pir::Operation* BuildKernelOp(
    const std::string& kernel_fn_str,
    const phi::KernelKey& kernel_key,
    const std::vector<pir::Value>& vec_inputs,
    const std::vector<pir::Type>& op_output_types,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  std::unordered_map<std::string, pir::Attribute> op_attribute{
      {"op_name", pir::StrAttribute::get(ctx, op_item->name())},
      {"kernel_name", pir::StrAttribute::get(ctx, kernel_fn_str)},
      {"kernel_key", KernelAttribute::get(ctx, kernel_key)}};
  auto op_attr_map = op_item->attributes();

  for (auto& map_item : op_attr_map) {
    op_attribute.emplace(map_item.first, map_item.second);
  }

  if (op_item->HasTrait<InplaceTrait>()) {
    op_attribute.emplace("is_inplace", pir::BoolAttribute::get(ctx, true));
  }

  op_attribute.emplace("origin_id",
                       pir::Int64Attribute::get(ctx, op_item->id()));

  pir::Operation* op = nullptr;
#ifdef PADDLE_WITH_DNNL
  if (op_item->HasTrait<OneDNNTrait>()) {
    auto op_info_parser = GetOpYamlInfoParser(op_item);
    std::vector<pir::Attribute> extra_args;
    for (auto& arg : op_info_parser->OpRuntimeInfo().extra_args) {
      extra_args.push_back(pir::StrAttribute::get(ctx, arg));
    }
    op_attribute.emplace(
        "extra_args",
        pir::ArrayAttribute::get(pir::IrContext::Instance(), extra_args));
    std::vector<pir::Attribute> skip_transform_inputs;
    for (auto& arg : op_info_parser->OpRuntimeInfo().skip_transform_inputs) {
      skip_transform_inputs.push_back(pir::StrAttribute::get(ctx, arg));
    }
    op_attribute.emplace("skip_transform_inputs",
                         pir::ArrayAttribute::get(pir::IrContext::Instance(),
                                                  skip_transform_inputs));
    std::vector<pir::Attribute> data_format_tensors;
    for (auto& input : op_info_parser->OpRuntimeInfo().data_format_tensors) {
      data_format_tensors.push_back(pir::StrAttribute::get(ctx, input));
    }
    op_attribute.emplace("data_format_tensors",
                         pir::ArrayAttribute::get(pir::IrContext::Instance(),
                                                  data_format_tensors));
    op_attribute.emplace(
        "is_onednn_only",
        pir::BoolAttribute::get(
            ctx, op_info_parser->OpRuntimeInfo().is_onednn_only));
    op_attribute.emplace(
        "dynamic_fallback",
        pir::BoolAttribute::get(
            ctx, op_info_parser->OpRuntimeInfo().dynamic_fallback));

    if (IsLegacyOp(op_item->name())) {
      VLOG(4) << "choose OneDNNLegacyKernelOp";
      pir::OpInfo legacy_kernel_op_info =
          ctx->GetRegisteredOpInfo(OneDNNLegacyKernelOp::name());
      op = pir::Operation::Create(
          vec_inputs, op_attribute, op_output_types, legacy_kernel_op_info);
    } else {
      if (op_item->HasTrait<OneDNNDynamicFallbackTrait>()) {
        VLOG(4) << "choose OneDNNMixedPhiKernelOp";
        pir::OpInfo phi_kernel_op_info =
            ctx->GetRegisteredOpInfo(OneDNNMixedPhiKernelOp::name());

        op = pir::Operation::Create(
            vec_inputs, op_attribute, op_output_types, phi_kernel_op_info);
      } else {
        VLOG(4) << "choose OneDNNPhiKernelOp";
        pir::OpInfo phi_kernel_op_info =
            ctx->GetRegisteredOpInfo(OneDNNPhiKernelOp::name());

        op = pir::Operation::Create(
            vec_inputs, op_attribute, op_output_types, phi_kernel_op_info);
      }
    }
  } else  // NOLINT
#endif
  {
    if (IsLegacyOp(op_item->name())) {
      pir::OpInfo legacy_kernel_op_info =
          ctx->GetRegisteredOpInfo(LegacyKernelOp::name());

      op = pir::Operation::Create(
          vec_inputs, op_attribute, op_output_types, legacy_kernel_op_info);
    } else {
      pir::OpInfo phi_kernel_op_info =
          ctx->GetRegisteredOpInfo(PhiKernelOp::name());
      op = pir::Operation::Create(
          vec_inputs, op_attribute, op_output_types, phi_kernel_op_info);
    }
  }
  op->set_attribute("origin_id", pir::Int64Attribute::get(ctx, op->id()));
  (*map_op_pair)[op_item] = op;
  // only deal with single output
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      (*map_value_pair)[op_item->result(i)] = op->result(i);
    }
  }
  block->push_back(op);

  return op;
}

#ifdef PADDLE_WITH_DNNL
pir::Operation* OneDNNOp2PdOp(pir::Operation* op_item,
                              pir::Block* block,
                              pir::IrContext* ctx) {
  std::vector<pir::Type> op_item_inner_output_types;
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      op_item_inner_output_types.push_back(op_item->result_type(i));
    }
  }
  std::string target_op_name = op_item->name();
  target_op_name.replace(0, 9, "pd_op");
  auto op_info = ctx->GetRegisteredOpInfo(target_op_name);
  if (!op_info) {
    IR_THROW("Ctx should have corresponding OpInfo %s", target_op_name);
  }
  pir::Operation* op_item_inner =
      pir::Operation::Create(op_item->operands_source(),
                             op_item->attributes(),
                             op_item_inner_output_types,
                             op_info);
  op_item_inner->set_attribute(
      "origin_id", pir::Int64Attribute::get(ctx, op_item_inner->id()));
  op_item->ReplaceAllUsesWith(op_item_inner->results());
  for (auto iter = block->begin(); iter != block->end(); ++iter) {  // NOLINT
    if (*iter == *op_item) {
      block->Assign(iter, op_item_inner);
      break;
    }
  }
  return op_item_inner;
}

pir::Operation* PdOp2OneDNNOp(pir::Operation* op_item,
                              pir::Block* block,
                              pir::IrContext* ctx) {
  std::string target_op_name = op_item->name();
  target_op_name.replace(0, 5, "onednn_op");
  auto op_info = ctx->GetRegisteredOpInfo(target_op_name);
  if (op_info) {
    std::vector<pir::Type> op_item_inner_output_types;
    if (op_item->num_results() > 0) {
      for (size_t i = 0; i < op_item->num_results(); ++i) {
        op_item_inner_output_types.push_back(op_item->result_type(i));
      }
    }
    auto attributes = op_item->attributes();
    auto yaml_interface =
        op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
    OpRunTimeInfo runtime_info =
        std::get<3>(yaml_interface->get_op_info_(target_op_name));
    for (auto& attr : runtime_info.extra_args_default_value) {
      attributes[attr.first] = attr.second;
    }
    pir::Operation* op_item_inner =
        pir::Operation::Create(op_item->operands_source(),
                               attributes,
                               op_item_inner_output_types,
                               op_info);
    op_item_inner->set_attribute(
        "origin_id", pir::Int64Attribute::get(ctx, op_item_inner->id()));
    op_item->ReplaceAllUsesWith(op_item_inner->results());
    for (auto iter = block->begin(); iter != block->end(); ++iter) {  // NOLINT
      if (*iter == *op_item) {
        block->Assign(iter, op_item_inner);
        break;
      }
    }
    return op_item_inner;
  } else {
    return op_item;
  }
}

#endif
void ProcessBlock(
    const phi::Place& place,
    pir::Block* block,
    pir::Block* new_block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair,
    bool for_if_block) {
  auto inputs_by_data_op = GetInputsByDataOp(block);
  for (auto& [keyword, arg] : block->kwargs()) {
    auto new_arg = new_block->AddKwarg(keyword, arg.type());
    const pir::BlockArgument& block_arg = arg.dyn_cast<pir::BlockArgument>();
    for (auto& [name, attr] : block_arg.attributes()) {
      new_arg.set_attribute(name, attr);
    }
    if (auto dense_tensor_type = arg.type().dyn_cast<DenseTensorType>()) {
      auto place_attr = arg.attribute<PlaceAttribute>("place");
      auto var_place = place_attr ? place_attr.data() : phi::Place();
      new_arg.set_type(
          AllocatedDenseTensorType::get(ctx, var_place, dense_tensor_type));
    }
    (*map_value_pair)[arg] = new_arg;
  }
  if (phi::is_accelerat_place(place)) {
    for (auto& [keyword, arg] : block->kwargs()) {
      if (auto dense_tensor_type = arg.type().dyn_cast<DenseTensorType>()) {
        auto place_attr = arg.attribute<PlaceAttribute>("place");
        if (place_attr && place_attr.data() == place) continue;
        auto dtype = dense_tensor_type.dtype();
        phi::KernelKey shadow_key{paddle::experimental::get_accelerat_backend(),
                                  phi::DataLayout::ANY,
                                  TransToPhiDataType(dtype)};
        std::unordered_map<std::string, pir::Attribute> attr_map{
            {"op_name", pir::StrAttribute::get(ctx, "pd_op.shadow_feed")},
            {"kernel_name", pir::StrAttribute::get(ctx, "shadow_feed")},
            {"kernel_key", KernelAttribute::get(ctx, shadow_key)},
            {"dst_place_type", pir::Int32Attribute::get(ctx, 1)}};

        auto out_type =
            AllocatedDenseTensorType::get(ctx, place, dense_tensor_type);

        pir::OpInfo phi_kernel_op_info =
            ctx->GetRegisteredOpInfo(PhiKernelOp::name());
        pir::Operation* shadow_op = pir::Operation::Create(
            {(*map_value_pair)[arg]}, attr_map, {out_type}, phi_kernel_op_info);
        shadow_op->set_attribute(
            "origin_id", pir::Int64Attribute::get(ctx, shadow_op->id()));

        new_block->push_back(shadow_op);
        (*map_value_pair)[arg] = shadow_op->result(0);
      } else if (auto vec_type = arg.type().dyn_cast<pir::VectorType>()) {
        for (size_t i = 0; i < vec_type.size(); ++i) {
          PADDLE_ENFORCE_EQ(
              vec_type[i].isa<DenseTensorType>(),
              true,
              common::errors::PreconditionNotMet(
                  "AddShadowFeedTensors only support DenseTensorType Now"));
        }
        // Add ShadowFeedTensors Op
        phi::KernelKey shadow_key{
            paddle::experimental::get_accelerat_backend(),
            phi::DataLayout::ANY,
            TransToPhiDataType(
                vec_type[0].dyn_cast<DenseTensorType>().dtype())};

        std::unordered_map<std::string, pir::Attribute> attr_map{
            {"op_name",
             pir::StrAttribute::get(ctx, "pd_op.shadow_feed_tensors")},
            {"kernel_name", pir::StrAttribute::get(ctx, "shadow_feed_tensors")},
            {"kernel_key", KernelAttribute::get(ctx, shadow_key)},
            {"dst_place_type", pir::Int32Attribute::get(ctx, 1)}};

        pir::OpInfo phi_kernel_op_info =
            ctx->GetRegisteredOpInfo(PhiKernelOp::name());

        std::vector<pir::Type> vec_out_types;
        for (size_t i = 0; i < vec_type.size(); ++i) {
          vec_out_types.push_back(AllocatedDenseTensorType::get(
              ctx,
              phi::TransToPhiPlace(shadow_key.backend()),
              vec_type[i].dyn_cast<DenseTensorType>()));
        }
        auto out_type = pir::VectorType::get(ctx, vec_out_types);
        pir::Operation* shadow_tensors_op = pir::Operation::Create(
            {(*map_value_pair)[arg]}, attr_map, {out_type}, phi_kernel_op_info);
        shadow_tensors_op->set_attribute(
            "origin_id",
            pir::Int64Attribute::get(ctx, shadow_tensors_op->id()));
        new_block->push_back(shadow_tensors_op);
        (*map_value_pair)[arg] = shadow_tensors_op->result(0);
      }
    }
  }

  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);
    VLOG(6) << "op name " << op_item->name();
    if ((op_item->isa<FeedOp>()) &&
        inputs_by_data_op.count(op_item->attributes()
                                    .at("name")
                                    .dyn_cast<pir::StrAttribute>()
                                    .AsString())) {
      VLOG(6) << "Skip FeedOp while lowering to kernel pass";
      continue;
    }

    // HandleSpecialOp
    if (SpecialLowerOps.count(op_item->name())) {
      VLOG(6) << "Handle Special Op: [" << op_item->name()
              << "] while lowering to kernel pass";
      HandleForSpecialOp(place,
                         op_item,
                         new_block,
                         ctx,
                         map_op_pair,
                         map_value_pair,
                         for_if_block);
      continue;
    }

    auto op_info_parser = GetOpYamlInfoParser(op_item);
    auto kernel_name = GetKernelName(op_info_parser.get(), op_item);
    auto kernel_key = GetKernelKey(
        op_item, place, kernel_name, *map_value_pair, op_info_parser.get());
    VLOG(6) << "kernel type " << kernel_key;

    if (paddle::dialect::IsCustomOp(op_item)) {
      HandleForCustomOp(ctx,
                        op_item,
                        kernel_key,
                        place,
                        op_info_parser.get(),
                        map_op_pair,
                        map_value_pair,
                        new_block);
      continue;
    }

    if (paddle::dialect::IsTensorRTOp(op_item)) {
      HandleForTensorRTOp(ctx,
                          op_item,
                          kernel_key,
                          place,
                          map_op_pair,
                          map_value_pair,
                          new_block);
      continue;
    }
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    if (paddle::dialect::IsCustomEngineOp(op_item)) {
      auto* interface = paddle::custom_engine::CustomEngineManager::Instance()
                            ->GetCustomEngineInterface();
      HandleForCustomEngineOp(ctx,
                              op_item,
                              &kernel_key,
                              place,
                              map_op_pair,
                              map_value_pair,
                              new_block,
                              interface);
      continue;
    }
#endif
#ifdef PADDLE_WITH_DNNL
    if (op_item->HasTrait<OneDNNTrait>() &&
        kernel_key.backend() != phi::Backend::ONEDNN) {
      auto op_item_inner = OneDNNOp2PdOp(op_item, block, ctx);
      op_item = op_item_inner;
      op_info_parser = GetOpYamlInfoParser(op_item_inner);
    }

    // Use OneDNN if CPU not support bf16
    if (kernel_key.dtype() == phi::DataType::BFLOAT16 &&
        kernel_key.backend() == phi::Backend::CPU &&
        !op_item->HasTrait<OneDNNTrait>() && !SupportsCPUBF16(kernel_name) &&
        SupportsMKLDNN(kernel_name, phi::DataType::BFLOAT16)) {
      auto op_item_inner = PdOp2OneDNNOp(op_item, block, ctx);
      if (op_item_inner != op_item) {
        op_item = op_item_inner;
        op_info_parser = GetOpYamlInfoParser(op_item_inner);
        kernel_key.set_backend(phi::Backend::ONEDNN);
        kernel_key.set_layout(phi::DataLayout::ONEDNN);
      }
    } else if (FLAGS_use_mkldnn && kernel_key.backend() == phi::Backend::CPU &&
               !op_item->HasTrait<OneDNNTrait>() &&
               SupportsMKLDNN(kernel_name, kernel_key.dtype())) {
      // Support FLAGS_use_mkldnn
      auto op_item_inner = PdOp2OneDNNOp(op_item, block, ctx);
      if (op_item_inner != op_item) {
        op_item = op_item_inner;
        op_info_parser = GetOpYamlInfoParser(op_item_inner);
        kernel_key.set_backend(phi::Backend::ONEDNN);
        kernel_key.set_layout(phi::DataLayout::ONEDNN);
      }
    } else if (kernel_key.backend() == phi::Backend::ONEDNN &&
               !op_item->HasTrait<OneDNNTrait>()) {
      auto op_item_inner = PdOp2OneDNNOp(op_item, block, ctx);
      if (op_item_inner != op_item) {
        op_item = op_item_inner;
        op_info_parser = GetOpYamlInfoParser(op_item_inner);
        kernel_key.set_backend(phi::Backend::ONEDNN);
        kernel_key.set_layout(phi::DataLayout::ONEDNN);
      }
    }
#endif
    // build input
    auto new_vec_inputs = BuildInputs(op_item,
                                      kernel_name,
                                      kernel_key,
                                      place,
                                      op_info_parser.get(),
                                      ctx,
                                      map_op_pair,
                                      map_value_pair,
                                      new_block);
    // build output type
    auto op_output_types =
        BuildOutputs(op_item, kernel_name, kernel_key, new_vec_inputs, ctx);

    // build op
    pir::Operation* op = BuildKernelOp(kernel_name,
                                       kernel_key,
                                       new_vec_inputs,
                                       op_output_types,
                                       op_item,
                                       new_block,
                                       ctx,
                                       map_op_pair,
                                       map_value_pair);

    AddShadowFeedOpForDataOrFeed(
        place, op_item, op, new_block, ctx, map_op_pair, map_value_pair);
  }

  RemoveRedundantMemcpyAfterShadowFeed(new_block, ctx);
}

std::unique_ptr<pir::Program> PdOpLowerToKernelPass(pir::Program* prog,
                                                    phi::Place place) {
  auto program = std::make_unique<pir::Program>(pir::IrContext::Instance());
  if (FLAGS_print_ir) {
    std::cout << "IR before lowering = " << *prog << std::endl;
  }
  auto block = prog->block();

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<KernelDialect>();
  ctx->GetOrRegisterDialect<CustomKernelDialect>();

#ifdef PADDLE_WITH_DNNL
  ctx->GetOrRegisterDialect<OneDNNOperatorDialect>();
  ctx->GetOrRegisterDialect<OneDNNKernelDialect>();
#endif
  std::unordered_map<pir::Operation*, pir::Operation*> map_op_pair;
  std::unordered_map<pir::Value, pir::Value> map_value_pair;

  ProcessBlock(
      place, block, program->block(), ctx, &map_op_pair, &map_value_pair);

  if (FLAGS_enable_collect_shape) {
    paddle::framework::CollectShapeManager::Instance().SetValueMap(
        map_value_pair);
  }

  if (FLAGS_print_ir) {
    std::cout << "IR after lowering = " << *program << std::endl;
  }

  return program;
}
}  // namespace paddle::dialect
