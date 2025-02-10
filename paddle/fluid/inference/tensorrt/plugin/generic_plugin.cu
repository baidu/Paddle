// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>

#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/inference/tensorrt/dynamic_shape_infermeta_registry.h"
#include "paddle/fluid/inference/tensorrt/plugin/generic_plugin.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/framework/framework.pb.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

GeneratePluginDataType ProtoTypeToGeneratePluginDataType(
    framework::proto::VarType_Type proto_type) {
  using framework::proto::VarType_Type;
  switch (proto_type) {
    case VarType_Type::VarType_Type_BOOL:
      return GeneratePluginDataType::PLUGIN_BOOL;
    case VarType_Type::VarType_Type_UINT8:
      return GeneratePluginDataType::PLUGIN_UINT8;
    case VarType_Type::VarType_Type_INT8:
      return GeneratePluginDataType::PLUGIN_INT8;
    case VarType_Type::VarType_Type_INT16:
      return GeneratePluginDataType::PLUGIN_INT16;
    case VarType_Type::VarType_Type_INT32:
      return GeneratePluginDataType::PLUGIN_INT32;
    case VarType_Type::VarType_Type_INT64:
      return GeneratePluginDataType::PLUGIN_INT64;
    case VarType_Type::VarType_Type_FP16:
      return GeneratePluginDataType::PLUGIN_FP16;
    case VarType_Type::VarType_Type_FP32:
      return GeneratePluginDataType::PLUGIN_FP32;
    case VarType_Type::VarType_Type_FP64:
      return GeneratePluginDataType::PLUGIN_FP64;
    case VarType_Type::VarType_Type_SIZE_T:
      return GeneratePluginDataType::PLUGIN_SIZE_T;
    case VarType_Type::VarType_Type_BF16:
      return GeneratePluginDataType::PLUGIN_BF16;
    case VarType_Type::VarType_Type_COMPLEX64:
      return GeneratePluginDataType::PLUGIN_COMPLEX64;
    case VarType_Type::VarType_Type_COMPLEX128:
      return GeneratePluginDataType::PLUGIN_COMPLEX128;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "This data type is currently not supported"));
  }
}

void BuildPhiKernelContextAttr(const framework::OpDesc& op_desc,
                               phi::KernelContext* kernel_context,
                               const phi::KernelSignature& signature,
                               const phi::Kernel* phi_kernel) {
  if (!phi_kernel->IsValid()) {
    return;
  }
  const phi::KernelArgsDef& args_def = phi_kernel->args_def();
  const auto& attr_names = signature.attr_names;
  const auto& attr_defs = args_def.attribute_defs();

  PADDLE_ENFORCE_EQ(
      attr_names.size(),
      attr_defs.size(),
      common::errors::InvalidArgument(
          "The attr_names.size() should be equal to attr_defs.size()."));

  framework::AttrReader attr_reader(op_desc.GetAttrMap());

  for (size_t k = 0; k < attr_names.size(); ++k) {
    auto attr_name = attr_names[k];
    auto* attr_ptr = attr_reader.GetAttr(attr_name);
    if (attr_ptr) {
      switch (attr_defs[k].type_index) {
        case phi::AttributeType::SCALAR: {
          auto& attr = *attr_ptr;
          switch (AttrTypeID(attr)) {
            case framework::proto::AttrType::FLOAT:
              kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(float, attr)));
              break;
            case framework::proto::AttrType::FLOAT64:
              kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(double, attr)));
              break;
            case framework::proto::AttrType::INT:
              kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(int, attr)));
              break;
            case framework::proto::AttrType::LONG:
              kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(int64_t, attr)));
              break;
            case framework::proto::AttrType::STRING:
              kernel_context->EmplaceBackAttr(
                  phi::Scalar(PADDLE_GET_CONST(std::string, attr)));
              break;
            case framework::proto::AttrType::SCALAR:
              kernel_context->EmplaceBackAttr(phi::Scalar(
                  PADDLE_GET_CONST(paddle::experimental::Scalar, attr)));
              break;
            default:
              PADDLE_THROW(common::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to Scalar when "
                  "ProtoAttr2PhiAttr.",
                  attr_name));
          }
        } break;

        case phi::AttributeType::INT_ARRAY: {
          auto& attr = *attr_ptr;
          switch (AttrTypeID(attr)) {
            case framework::proto::AttrType::INTS:
              kernel_context->EmplaceBackAttr(std::move(
                  phi::IntArray(PADDLE_GET_CONST(std::vector<int32_t>, attr))));
              break;
            case framework::proto::AttrType::LONGS:
              kernel_context->EmplaceBackAttr(std::move(
                  phi::IntArray(PADDLE_GET_CONST(std::vector<int64_t>, attr))));
              break;
            case framework::proto::AttrType::INT:
              kernel_context->EmplaceBackAttr(
                  phi::IntArray({PADDLE_GET_CONST(int, attr)}));
              break;
            default:
              PADDLE_THROW(common::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to IntArray when "
                  "ProtoAttr2PhiAttr.",
                  attr_name));
          }
        } break;

        case phi::AttributeType::SCALARS: {
          auto& attr = *attr_ptr;
          switch (AttrTypeID(attr)) {
            case framework::proto::AttrType::INTS: {
              const auto& vec = PADDLE_GET_CONST(std::vector<int32_t>, attr);
              std::vector<phi::Scalar> scalar_list;
              scalar_list.reserve(vec.size());
              for (const auto& val : vec) {
                scalar_list.emplace_back(val);
              }
              kernel_context->EmplaceBackAttr(std::move(scalar_list));
            } break;
            case framework::proto::AttrType::LONGS: {
              const auto& vec = PADDLE_GET_CONST(std::vector<int64_t>, attr);
              std::vector<phi::Scalar> scalar_list;
              scalar_list.reserve(vec.size());
              for (const auto& val : vec) {
                scalar_list.emplace_back(val);
              }
              kernel_context->EmplaceBackAttr(std::move(scalar_list));
            } break;
            case framework::proto::AttrType::FLOATS: {
              const auto& vec = PADDLE_GET_CONST(std::vector<float>, attr);
              std::vector<phi::Scalar> scalar_list;
              scalar_list.reserve(vec.size());
              for (const auto& val : vec) {
                scalar_list.emplace_back(val);
              }
              kernel_context->EmplaceBackAttr(std::move(scalar_list));
            } break;
            case framework::proto::AttrType::FLOAT64S: {
              const auto& vec = PADDLE_GET_CONST(std::vector<double>, attr);
              std::vector<phi::Scalar> scalar_list;
              scalar_list.reserve(vec.size());
              for (const auto& val : vec) {
                scalar_list.emplace_back(val);
              }
              kernel_context->EmplaceBackAttr(std::move(scalar_list));
            } break;
            case framework::proto::AttrType::SCALARS: {
              const auto& vec = PADDLE_GET_CONST(
                  std::vector<paddle::experimental::Scalar>, attr);
              std::vector<phi::Scalar> scalar_list;
              scalar_list.reserve(vec.size());
              for (const auto& val : vec) {
                scalar_list.emplace_back(val);
              }
              kernel_context->EmplaceBackAttr(std::move(scalar_list));
            } break;
            default:
              PADDLE_THROW(common::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to vector<Scalar> when "
                  "ProtoAttr2PhiAttr.",
                  attr_name));
          }
        } break;

        default: {
          auto& attr = *attr_ptr;
          switch (attr_defs[k].type_index) {
            case phi::AttributeType::FLOAT32:
              kernel_context->EmplaceBackAttr(PADDLE_GET_CONST(float, attr));
              break;
            case phi::AttributeType::FLOAT64:
              kernel_context->EmplaceBackAttr(PADDLE_GET_CONST(double, attr));
              break;
            case phi::AttributeType::INT32:
              kernel_context->EmplaceBackAttr(PADDLE_GET_CONST(int, attr));
              break;
            case phi::AttributeType::BOOL:
              kernel_context->EmplaceBackAttr(PADDLE_GET_CONST(bool, attr));
              break;
            case phi::AttributeType::INT64:
              kernel_context->EmplaceBackAttr(PADDLE_GET_CONST(int64_t, attr));
              break;
            case phi::AttributeType::INT32S:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::vector<int>, attr));
              break;
            case phi::AttributeType::DATA_TYPE: {
              auto data_type = phi::TransToPhiDataType(
                  static_cast<framework::proto::VarType::Type>(
                      PADDLE_GET_CONST(int, attr)));
              kernel_context->EmplaceBackAttr(data_type);
            } break;
            case phi::AttributeType::STRING:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::string, attr));
              break;
            case phi::AttributeType::INT64S:
              switch (AttrTypeID(attr)) {
                case framework::proto::AttrType::LONGS:
                  kernel_context->EmplaceBackAttr(
                      PADDLE_GET_CONST(std::vector<int64_t>, attr));
                  break;
                case framework::proto::AttrType::INTS: {
                  const auto& vector_int_attr =
                      PADDLE_GET_CONST(std::vector<int>, attr);
                  const std::vector<int64_t> vector_int64_attr(
                      vector_int_attr.begin(), vector_int_attr.end());
                  kernel_context->EmplaceBackAttr(vector_int64_attr);
                } break;
                default:
                  PADDLE_THROW(common::errors::Unimplemented(
                      "Unsupported cast op attribute `%s` to vector<int64_t> "
                      "when ProtoAttr2PhiAttr.",
                      attr_name));
              }
              break;
            case phi::AttributeType::FLOAT32S:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::vector<float>, attr));
              break;
            case phi::AttributeType::STRINGS:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::vector<std::string>, attr));
              break;
            case phi::AttributeType::BOOLS:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::vector<bool>, attr));
              break;
            case phi::AttributeType::FLOAT64S:
              kernel_context->EmplaceBackAttr(
                  PADDLE_GET_CONST(std::vector<double>, attr));
              break;
            default:
              PADDLE_THROW(common::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` when construct "
                  "ProtoAttr2PhiAttr.",
                  attr_name));
          }
        }
      }
    }
  }

  PADDLE_ENFORCE_EQ(attr_names.size(),
                    kernel_context->AttrsSize(),
                    common::errors::InvalidArgument(
                        "The attr_names.size() should be equal to "
                        "kernel_context->AttrsSize()."
                        "Received attr_names.size() = % d,"
                        "kernel_context->AttrsSize() = %d.",
                        attr_names.size(),
                        kernel_context->AttrsSize()));
}

GenericPlugin::GenericPlugin(
    const paddle::framework::proto::OpDesc& proto_op_desc,
    const InputOutPutVarInfo& in_out_info,
    bool with_fp16) {
  proto_op_desc_ = proto_op_desc;
  op_desc_ = std::move(framework::OpDesc(proto_op_desc_, nullptr));
  proto_op_desc_.SerializeToString(&op_meta_data_);
  inputs_data_type_ = in_out_info.inputs_data_type;
  outputs_data_type_ = in_out_info.outputs_data_type;
  with_fp16_ = with_fp16;
}

GenericPlugin::GenericPlugin(
    const paddle::framework::proto::OpDesc& proto_op_desc,
    const std::vector<GeneratePluginDataType>& inputs_data_type,
    const std::vector<GeneratePluginDataType>& outputs_data_type,
    bool with_fp16) {
  proto_op_desc_ = proto_op_desc;
  op_desc_ = std::move(framework::OpDesc(proto_op_desc_, nullptr));
  proto_op_desc_.SerializeToString(&op_meta_data_);
  inputs_data_type_ = inputs_data_type;
  outputs_data_type_ = outputs_data_type;
  with_fp16_ = with_fp16;
}

GenericPlugin::GenericPlugin(void const* serial_data, size_t serial_length) {
  DeserializeValue(&serial_data, &serial_length, &inputs_data_type_);
  DeserializeValue(&serial_data, &serial_length, &outputs_data_type_);
  DeserializeValue(&serial_data, &serial_length, &with_fp16_);

  std::string op_meta_data((char*)(serial_data), serial_length);  // NOLINT
  op_meta_data_ = std::move(op_meta_data);
  proto_op_desc_.ParseFromString(op_meta_data_);
  op_desc_ = std::move(framework::OpDesc(proto_op_desc_, nullptr));
}

int GenericPlugin::getNbOutputs() const TRT_NOEXCEPT {
  int res = 0;
  for (auto& i : op_desc_.Outputs()) {
    if (!i.second.empty()) res += i.second.size();
  }
  return res;
}

int GenericPlugin::getNbInputs() const TRT_NOEXCEPT {
  int res = 0;
  for (auto& i : op_desc_.Inputs()) {
    if (!i.second.empty()) res += i.second.size();
  }
  return res;
}

nvinfer1::IPluginV2DynamicExt* GenericPlugin::clone() const TRT_NOEXCEPT {
  nvinfer1::IPluginV2DynamicExt* plugin = new GenericPlugin(
      proto_op_desc_, inputs_data_type_, outputs_data_type_, with_fp16_);
  plugin->initialize();
  return plugin;
}

void GenericPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
  // inputs_data_type_
  SerializeValue(&buffer, inputs_data_type_);
  // outputs_data_type_
  SerializeValue(&buffer, outputs_data_type_);
  // use fp16
  SerializeValue(&buffer, with_fp16_);
  // serialize op_meta_data_
  std::memcpy(buffer, op_meta_data_.c_str(), op_meta_data_.size());
  reinterpret_cast<char*&>(buffer) += op_meta_data_.size();
}

bool GenericPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  if (op_desc_.Type() == "gather_nd" || op_desc_.Type() == "yolo_box") {
    if (pos == 0)
      return (in_out[pos].type == nvinfer1::DataType::kFLOAT ||
              (isFp16Supported() &&
               in_out[pos].type == nvinfer1::DataType::kHALF)) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    if (pos == 1)
      return (in_out[pos].type == nvinfer1::DataType::kINT32) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // output
    if (pos == 2 || pos == 3)
      return in_out[0].type == in_out[pos].type &&
             in_out[0].format == in_out[pos].format;
  } else if (op_desc_.Type() == "scatter_nd_add") {
    // input X
    if (pos == 0)
      return (in_out[pos].type == nvinfer1::DataType::kFLOAT ||
              (isFp16Supported() &&
               in_out[pos].type == nvinfer1::DataType::kHALF)) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // input Index
    if (pos == 1)
      return (in_out[pos].type == nvinfer1::DataType::kINT32) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // input Updates and output
    if (pos == 2 || pos == 3)
      return in_out[0].type == in_out[pos].type &&
             in_out[0].format == in_out[pos].format;
  } else if (op_desc_.Type() == "lookup_table_v2") {
    if (pos == 0)
      return (in_out[pos].type == nvinfer1::DataType::kINT32 &&
              (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR));
    if (pos == 1)
      return (in_out[pos].type == nvinfer1::DataType::kFLOAT) ||
             ((isFp16Supported() &&
               in_out[pos].type == nvinfer1::DataType::kHALF)) &&
                 (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // output
    if (pos == 2)
      return in_out[1].type == in_out[pos].type &&
             in_out[1].format == in_out[pos].format;
  } else if (op_desc_.Type() == "argsort") {
    // input x
    if (pos == 0) {
      return ((in_out[pos].type == nvinfer1::DataType::kFLOAT ||
               (isFp16Supported() &&
                in_out[pos].type == nvinfer1::DataType::kHALF)) &&
              in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    }
    // output out
    if (pos == 1) {
      return (in_out[pos].type == in_out[0].type &&
              in_out[pos].format == in_out[0].format);
    }
    // output indices
    if (pos == 2) {
      return (in_out[pos].type == nvinfer1::DataType::kINT32 &&
              in_out[pos].format == in_out[0].format);
    }
  } else if (op_desc_.Type() == "scatter") {
    // input X
    if (pos == 0)
      return (in_out[pos].type == nvinfer1::DataType::kFLOAT ||
              (isFp16Supported() &&
               in_out[pos].type == nvinfer1::DataType::kHALF)) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // Ids
    if (pos == 1)
      return (in_out[pos].type == nvinfer1::DataType::kINT32) &&
             (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR);
    // 3:output 2:input Updates
    if (pos == 3 || pos == 2)
      return in_out[0].type == in_out[pos].type &&
             in_out[0].format == in_out[pos].format;
  } else if (op_desc_.Type() == "solve") {
    // input X
    if (pos == 0)
      return in_out[pos].type == nvinfer1::DataType::kFLOAT &&
             in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
    // input Y
    if (pos == 1)
      return in_out[pos].type == nvinfer1::DataType::kFLOAT &&
             in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
    // output
    if (pos == 2)
      return in_out[0].type == in_out[pos].type &&
             in_out[0].format == in_out[pos].format;
  } else {
    return (in_out[pos].type == nvinfer1::DataType::kFLOAT ||
            (isFp16Supported() &&
             in_out[pos].type == nvinfer1::DataType::kHALF)) &&
           (in_out[pos].format == nvinfer1::TensorFormat::kLINEAR) &&
           (in_out[0].type == in_out[pos].type);
  }
}

nvinfer1::DataType GenericPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  if (op_desc_.Type() == "lookup_table_v2") {
    return input_types[1];
  }
  if (op_desc_.Type() == "argsort") {
    if (index == 1) {
      return nvinfer1::DataType::kINT32;
    }
  }
  return input_types[0];
}

int GenericPlugin::initialize() TRT_NOEXCEPT {
  std::string op_type = op_desc_.Type();

  phi::KernelSignature phi_kernel_signature;
  if (phi::OpUtilsMap::Instance().HasArgumentMappingFn(op_type)) {
    const phi::ArgumentMappingFn* argument_mapping_func =
        phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_type);
    PluginArgumentMappingContext argument_mapping_context(&op_desc_);
    phi_kernel_signature = (*argument_mapping_func)(argument_mapping_context);
  } else {
    phi_kernel_signature =
        phi::DefaultKernelSignatureMap::Instance().Get(op_type);
  }

  PADDLE_ENFORCE_EQ(
      phi::KernelFactory::Instance().HasCompatiblePhiKernel(op_type),
      true,
      common::errors::Fatal("%s has no compatible phi kernel!",
                            op_type.c_str()));

  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  phi::GPUPlace place(platform::GetCurrentDeviceId());
  auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(place));

  std::vector<phi::DataType> precision_types{phi::DataType::FLOAT32,
                                             phi::DataType::FLOAT16};
  for (auto& precision_type : precision_types) {
    phi::KernelKey phi_kernel_key(
        phi::Backend::GPU, phi::DataLayout::ANY, precision_type);

    auto nv_dtype = PhiType2NvType(precision_type);
    phi_kernels_[nv_dtype] = std::make_unique<phi::Kernel>(
        phi::KernelFactory::Instance().SelectKernel(phi_kernel_signature.name,
                                                    phi_kernel_key));

    if (phi_kernel_contexts_.find(nv_dtype) == phi_kernel_contexts_.end() ||
        !phi_kernel_contexts_[nv_dtype]) {
      phi_kernel_contexts_[nv_dtype] =
          std::make_unique<phi::KernelContext>(dev_ctx);
      BuildPhiKernelContextAttr(op_desc_,
                                phi_kernel_contexts_[nv_dtype].get(),
                                phi_kernel_signature,
                                phi_kernels_[nv_dtype].get());
    }
  }
  PADDLE_ENFORCE_EQ(phi_kernels_[nvinfer1::DataType::kFLOAT]->IsValid() ||
                        phi_kernels_[nvinfer1::DataType::kHALF]->IsValid(),
                    true,
                    common::errors::Fatal("%s phi kernel is invalid!.",
                                          phi_kernel_signature.name));

  if (!dense_tensor_inputs_)
    dense_tensor_inputs_ = new std::vector<phi::DenseTensor>(getNbInputs());
  if (!dense_tensor_outputs_)
    dense_tensor_outputs_ = new std::vector<phi::DenseTensor>(getNbOutputs());

  return 0;
}

nvinfer1::DimsExprs GenericPlugin::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      output_index < getNbOutputs(),
      true,
      common::errors::InvalidArgument(
          "The output_index should be less than getNbOutputs()."));
  auto& dynamic_infermeta_factory = tensorrt::DynamicMetaFnFactory::Instance();
  PADDLE_ENFORCE_EQ(dynamic_infermeta_factory.Contains(op_desc_.Type()),
                    true,
                    common::errors::InvalidArgument(
                        "The %s op has no dynamic plugin infershape function!",
                        op_desc_.Type().c_str()));

  auto* infershape_func = dynamic_infermeta_factory.Get(op_desc_.Type());
  return infershape_func(
      output_index, inputs, nb_inputs, expr_builder, op_desc_);
}

void GenericPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(phi_kernels_[nvinfer1::DataType::kFLOAT]->IsValid() ||
                        phi_kernels_[nvinfer1::DataType::kHALF]->IsValid(),
                    true,
                    common::errors::Fatal("Sorry, phi kernel is invalid!"));
  PADDLE_ENFORCE_EQ(nb_inputs == getNbInputs(),
                    true,
                    common::errors::InvalidArgument(
                        "The nb_inputs should be equal to getNbInputs()."));
  PADDLE_ENFORCE_EQ(nb_outputs == getNbOutputs(),
                    true,
                    common::errors::InvalidArgument(
                        "The nb_outputs should be equal to getNbOutputs()."));
}

// Shutdown the layer. This is called when the engine is destroyed
void GenericPlugin::terminate() TRT_NOEXCEPT {
  delete dense_tensor_inputs_;
  delete dense_tensor_outputs_;
}

int GenericPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                           const nvinfer1::PluginTensorDesc* output_desc,
                           const void* const* inputs,
                           void* const* outputs,
                           void* workspace,
                           cudaStream_t stream) TRT_NOEXCEPT {
  phi::GPUPlace place(platform::GetCurrentDeviceId());
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  // TODO(inference): generic plugin do not support INT8 precision now.
  auto nvType2PhiType =
      [&](nvinfer1::DataType nv_dtype) -> std::pair<phi::DataType, int> {
    const std::map<nvinfer1::DataType, std::pair<phi::DataType, int>> _map{
        {nvinfer1::DataType::kFLOAT, {phi::DataType::FLOAT32, sizeof(float)}},
        {nvinfer1::DataType::kHALF, {phi::DataType::FLOAT16, sizeof(half)}},
        {nvinfer1::DataType::kINT32, {phi::DataType::INT32, sizeof(int32_t)}},
        {nvinfer1::DataType::kBOOL, {phi::DataType::BOOL, sizeof(bool)}},
    };
    PADDLE_ENFORCE_EQ(
        _map.count(nv_dtype),
        true,
        common::errors::InvalidArgument("Sorry, dtype [ %d ] is not supported.",
                                        static_cast<int>(nv_dtype)));
    return _map.at(nv_dtype);
  };

  nvinfer1::DataType data_type;
  // input
  if (op_desc_.Type() == "lookup_table_v2") {
    data_type = input_desc[1].type;
  } else {
    data_type = input_desc[0].type;
  }
  PADDLE_ENFORCE_EQ((data_type == nvinfer1::DataType::kFLOAT) ||
                        (data_type == nvinfer1::DataType::kHALF),
                    true,
                    common::errors::InvalidArgument(
                        "The data_type should be kFLOAT or kHALF."));
  phi_kernel_contexts_[data_type]->ClearInputOutput();

  auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(place));
  phi_kernel_contexts_[data_type]->SetDeviceContext(dev_ctx);

  for (int i = 0; i < getNbInputs(); i++) {
    if (inputs_data_type_[i] == GeneratePluginDataType::PLUGIN_OPTIONAL) {
      phi_kernel_contexts_[data_type]->EmplaceBackInput(nullptr);
      continue;
    }
    auto const& input_dims = input_desc[i].dims;

    std::vector<int> input_shape;
    for (int j = 0; j < input_dims.nbDims; j++)
      input_shape.push_back(input_dims.d[j]);

    int input_numel = 1;
    for (int k = 0; k < input_shape.size(); k++) input_numel *= input_shape[k];
    auto data_type_and_size = nvType2PhiType(input_desc[i].type);
    phi::DenseTensorMeta input_meta(data_type_and_size.first,
                                    common::make_ddim(input_shape));
    std::shared_ptr<phi::Allocation> input_alloc(
        new phi::Allocation((void*)(inputs[i]),  // NOLINT
                            input_numel * data_type_and_size.second,
                            place));
    (*dense_tensor_inputs_)[i] =
        std::move(phi::DenseTensor(input_alloc, input_meta));
    phi_kernel_contexts_[data_type]->EmplaceBackInput(
        &((*dense_tensor_inputs_)[i]));
  }
  // output
  for (int i = 0; i < getNbOutputs(); i++) {
    auto const& output_dims = output_desc[i].dims;

    std::vector<int> output_shape;
    for (int j = 0; j < output_dims.nbDims; j++)
      output_shape.push_back(output_dims.d[j]);

    int output_numel = 1;
    for (int k = 0; k < output_shape.size(); k++)
      output_numel *= output_shape[k];

    auto data_type_and_size = nvType2PhiType(output_desc[i].type);
    phi::DenseTensorMeta output_meta(data_type_and_size.first,
                                     common::make_ddim(output_shape));
    std::shared_ptr<phi::Allocation> output_alloc(
        new phi::Allocation(reinterpret_cast<void*>(outputs[i]),
                            output_numel * data_type_and_size.second,
                            place));

    (*dense_tensor_outputs_)[i] =
        std::move(phi::DenseTensor(output_alloc, output_meta));

    phi_kernel_contexts_[data_type]->EmplaceBackOutput(
        &((*dense_tensor_outputs_)[i]));
  }

  PADDLE_ENFORCE_EQ(
      phi_kernel_contexts_[data_type]->InputsSize(),
      getNbInputs(),
      common::errors::InvalidArgument(
          "The phi_kernel_contexts_[data_type]->InputsSize() "
          "should be equal to getNbInputs()."
          "Received phi_kernel_contexts_[data_type]->InputsSize() "
          "= %d, getNbInputs() = %d.",
          phi_kernel_contexts_[data_type]->InputsSize()));
  PADDLE_ENFORCE_EQ(phi_kernel_contexts_[data_type]->OutputsSize(),
                    getNbOutputs(),
                    common::errors::InvalidArgument(
                        "The phi_kernel_contexts_[data_type]->OutputsSize() "
                        "should be equal to getNbOutputs()."));
  (*phi_kernels_[data_type])(phi_kernel_contexts_[data_type].get());

  if (op_desc_.Type() == "argsort") {
    for (int i = 0; i < getNbOutputs(); i++) {
      phi::DenseTensor& output_tensor = (*dense_tensor_outputs_)[i];
      phi::DataType dtype = output_tensor.dtype();
      if (dtype == phi::DataType::INT64) {
        auto& int32_tensor = output_tensor;
        auto ctx = pool.Get(output_tensor.place());
        int32_tensor = phi::funcs::TransDataType(
            reinterpret_cast<const phi::GPUContext&>(*ctx),
            output_tensor,
            phi::DataType::INT32);
        paddle::memory::Copy(output_tensor.place(),
                             outputs[i],
                             output_tensor.place(),
                             int32_tensor.data<int32_t>(),
                             int32_tensor.numel() * sizeof(int),
                             nullptr);
      }
    }
  }

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
