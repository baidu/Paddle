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

#include "paddle/fluid/framework/phi_utils.h"

#include <sstream>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/type_defs.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle::framework {

class KernelArgsNameMakerByOpProto : public KernelArgsNameMaker {
 public:
  explicit KernelArgsNameMakerByOpProto(
      const framework::proto::OpProto* op_proto)
      : op_proto_(op_proto), input_names_(), output_names_(), attr_names_() {
    PADDLE_ENFORCE_NOT_NULL(
        op_proto_,
        common::errors::InvalidArgument("Op proto cannot be nullptr."));
  }

  ~KernelArgsNameMakerByOpProto() override = default;

  const paddle::small_vector<const char*>& GetInputArgsNames() override;
  const paddle::small_vector<const char*>& GetOutputArgsNames() override;
  const paddle::small_vector<const char*>& GetAttrsArgsNames() override;

  phi::KernelSignature GetKernelSignature();

 private:
  DISABLE_COPY_AND_ASSIGN(KernelArgsNameMakerByOpProto);

 private:
  const framework::proto::OpProto* op_proto_;

  paddle::small_vector<const char*> input_names_;
  paddle::small_vector<const char*> output_names_;
  paddle::small_vector<const char*> attr_names_;
};

OpKernelType TransPhiKernelKeyToOpKernelType(const phi::KernelKey& kernel_key) {
  proto::VarType::Type data_type =
      paddle::framework::TransToProtoVarType(kernel_key.dtype());
  // no need to set current device id here
  phi::Place place = phi::TransToPhiPlace(kernel_key.backend(), false);
  DataLayout data_layout = kernel_key.layout();
  LibraryType library_type = LibraryType::kPlain;
  if (kernel_key.backend() == phi::Backend::ONEDNN) {
    library_type = LibraryType::kMKLDNN;
  } else if (kernel_key.backend() == phi::Backend::GPUDNN) {
    library_type = LibraryType::kCUDNN;
  } else if (kernel_key.backend() == phi::Backend::KPS) {
    library_type = LibraryType::kKP;
  } else {
    // do nothing
  }
  // TODO(chenweihang): the customized_type_value is lost
  return OpKernelType(data_type, place, data_layout, library_type);
}

phi::KernelKey TransOpKernelTypeToPhiKernelKey(
    const OpKernelType& kernel_type) {
  phi::Backend backend = phi::TransToPhiBackend(kernel_type.place_);
  switch (kernel_type.library_type_) {
    case LibraryType::kCUDNN:
      backend = phi::Backend::GPUDNN;
      break;
    case LibraryType::kMKLDNN:
      backend = phi::Backend::ONEDNN;
      break;
    case LibraryType::kKP:
      backend = phi::Backend::KPS;
      break;
    default:
      break;
  }
  return phi::KernelKey(backend,
                        kernel_type.data_layout_,
                        phi::TransToPhiDataType(kernel_type.data_type_));
}

phi::KernelKey FallBackToCpu(const phi::KernelKey& kernel_key,
                             const framework::OperatorBase& op) {
#ifdef PADDLE_WITH_XPU
  if (kernel_key.backend() == phi::Backend::XPU ||
      paddle::platform::is_in_xpu_black_list(op.Type())) {
    VLOG(3) << "phi missing XPU kernel: " << op.Type()
            << ", expected_kernel_key:" << kernel_key
            << ", fallback to CPU one!";
    return phi::KernelKey(
        phi::Backend::CPU, kernel_key.layout(), kernel_key.dtype());
  }
#endif
#ifdef PADDLE_WITH_IPU
  if (kernel_key.backend() == phi::Backend::IPU) {
    VLOG(3) << "phi missing IPU kernel: " << op.Type()
            << ", expected_kernel_key:" << kernel_key
            << ", fallback to CPU one!";
    return phi::KernelKey(
        phi::Backend::CPU, kernel_key.layout(), kernel_key.dtype());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  auto place = phi::TransToPhiPlace(kernel_key.backend());
  bool is_custom_place = phi::is_custom_place(place);
  if (is_custom_place ||
      phi::backends::custom_device::is_in_custom_black_list(op.Type())) {
    std::string info = is_custom_place ? "phi missing " : "phi in black list ";
    VLOG(3) << info << place.GetDeviceType() << " kernel: " << op.Type()
            << ", expected_kernel_key:" << kernel_key
            << ", fallback to CPU one!";
    return phi::KernelKey(
        phi::Backend::CPU, kernel_key.layout(), kernel_key.dtype());
  }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (kernel_key.backend() == phi::Backend::GPU ||
      kernel_key.backend() == phi::Backend::GPUDNN) {
    PADDLE_THROW(
        common::errors::NotFound("The kernel (%s) with key %s is not found and "
                                 "GPU kernel cannot fallback to CPU one.",
                                 op.Type(),
                                 kernel_key));
  }
#endif

  return phi::KernelKey();
}

const paddle::small_vector<const char*>&
KernelArgsNameMakerByOpProto::GetInputArgsNames() {
  for (int i = 0; i < op_proto_->inputs_size(); ++i) {
    auto& in = op_proto_->inputs()[i];
    auto& in_name = in.name();
    if ((in.has_extra() && in.extra()) || (in.has_quant() && in.quant())) {
      continue;
    }
    // If contains dispensable input, we should override the
    // OpArgumentMapping method self in fluid/operators/ops_signature dir
    if (in.has_dispensable() && in.dispensable()) {
      continue;
    }
    input_names_.emplace_back(in_name.c_str());
  }
  if (VLOG_IS_ON(10)) {
    std::ostringstream sout;
    sout << "PhiKernel inputs: ";
    std::copy(input_names_.begin(),
              input_names_.end(),
              std::ostream_iterator<const char*>(sout, ", "));
    VLOG(10) << sout.str();
  }
  return input_names_;
}

const paddle::small_vector<const char*>&
KernelArgsNameMakerByOpProto::GetOutputArgsNames() {
  for (int i = 0; i < op_proto_->outputs_size(); ++i) {
    auto& out = op_proto_->outputs()[i];
    auto& out_name = out.name();
    if ((out.has_extra() && out.extra()) || (out.has_quant() && out.quant())) {
      continue;
    }
    output_names_.emplace_back(out_name.c_str());
  }
  if (VLOG_IS_ON(10)) {
    std::ostringstream sout;
    sout << "PhiKernel outputs: ";
    std::copy(output_names_.begin(),
              output_names_.end(),
              std::ostream_iterator<const char*>(sout, ", "));
    VLOG(10) << sout.str();
  }
  return output_names_;
}

const paddle::small_vector<const char*>&
KernelArgsNameMakerByOpProto::GetAttrsArgsNames() {
  for (int i = 0; i < op_proto_->attrs_size(); ++i) {
    auto& attr = op_proto_->attrs()[i];
    auto& attr_name = attr.name();
    if (attr_name == "use_mkldnn" || attr_name == "use_cudnn" ||
        attr_name == "op_role" || attr_name == "op_role_var" ||
        attr_name == "op_namescope" || attr_name == "op_callstack" ||
        attr_name == "op_device") {
      continue;
    }
    if ((attr.has_extra() && attr.extra()) ||
        (attr.has_quant() && attr.quant())) {
      continue;
    }
    attr_names_.emplace_back(attr_name.c_str());
  }
  if (VLOG_IS_ON(10)) {
    std::ostringstream sout;
    sout << "PhiKernel attributes: ";
    std::copy(attr_names_.begin(),
              attr_names_.end(),
              std::ostream_iterator<const char*>(sout, ", "));
    VLOG(10) << sout.str();
  }
  return attr_names_;
}

phi::KernelSignature KernelArgsNameMakerByOpProto::GetKernelSignature() {
  return phi::KernelSignature(
      phi::TransToPhiKernelName(op_proto_->type()).c_str(),
      GetInputArgsNames(),
      GetAttrsArgsNames(),
      GetOutputArgsNames());
}

std::once_flag kernel_sig_map_init_flag;

void InitDefaultKernelSignatureMap() {
  std::call_once(kernel_sig_map_init_flag, [] {
    for (const auto& pair : paddle::framework::OpInfoMap::Instance().map()) {
      const auto& op_type = pair.first;
      const auto* op_proto = pair.second.proto_;
      if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(op_type) &&
          op_proto) {
        paddle::framework::KernelArgsNameMakerByOpProto maker(op_proto);
        VLOG(10) << "Register `" << op_type << "` kernel signature:";
        phi::DefaultKernelSignatureMap::Instance().Insert(
            op_type, maker.GetKernelSignature());
      }
    }
  });
}

static void SetAllocationForUninitializedDenseTensor(
    phi::DenseTensor* dense_tensor, const phi::Place& place) {
  int dtype_size = static_cast<int>(dense_tensor->dtype() == DataType::UNDEFINED
                                        ? 0
                                        : phi::SizeOf(dense_tensor->dtype()));
  int64_t numels = product(dense_tensor->dims());
  numels = numels < 0 ? 0 : numels;
  auto tmp_allocation_ptr = memory::Alloc(place, numels * dtype_size);
  auto& deleter = tmp_allocation_ptr.get_deleter();
  auto* allocation_ptr = tmp_allocation_ptr.release();
  auto shared_allocation =
      std::shared_ptr<phi::Allocation>(allocation_ptr, deleter);

  dense_tensor->ResetHolder(shared_allocation);
}

phi::Scalar MakePhiScalarFromVar(const framework::Variable& variable) {
  auto expected_place = phi::TransToPhiPlace(phi::Backend::CPU);
  if (variable.IsType<phi::DenseTensor>()) {
    const auto& tensor = variable.Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(
        tensor.numel(),
        1UL,
        common::errors::InvalidArgument("The DenseTensor used to construct "
                                        "the Scalar contains more than 1 "
                                        "value, it contains `%d` values.",
                                        tensor.numel()));
    if (!phi::is_same_place(tensor.place(), expected_place)) {
      phi::DenseTensor tmp_tensor;
      framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
      return {tmp_tensor};
    } else {
      return {tensor};
    }
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported casting input `%s` type to Scalar when call pt "
        "kernel.",
        framework::ToTypeName(variable.Type())));
  }
}

phi::IntArray MakePhiIntArrayFromVar(const framework::Variable& variable) {
  if (variable.IsType<phi::DenseTensor>()) {
    const auto& tensor = variable.Get<phi::DenseTensor>();
    return phi::IntArray(tensor);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported casting input `%s` type to IntArray when call pt "
        "kernel.",
        framework::ToTypeName(variable.Type())));
  }
}

// TODO(chentianyu03): Inplace with IntArray constructor
phi::IntArray MakePhiIntArrayFromVarList(
    const std::vector<framework::Variable*>& variable_list) {
  if (variable_list.empty()) {
    return phi::IntArray();
  }
  auto expected_place = phi::TransToPhiPlace(phi::Backend::CPU);

  std::vector<int64_t> vector_data;
  vector_data.reserve(variable_list.size());

  for (auto* var : variable_list) {
    phi::DataType data_type;
    if (var->IsType<phi::DenseTensor>()) {
      const auto& tensor = var->Get<phi::DenseTensor>();
      data_type = tensor.dtype();
      if (data_type == phi::DataType::INT64) {
        const auto& tensor = var->Get<phi::DenseTensor>();
        if (tensor.IsInitialized() &&
            !phi::is_same_place(tensor.place(), expected_place)) {
          phi::DenseTensor tmp_tensor;
          framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
          vector_data.push_back(*tmp_tensor.data<int64_t>());
        } else {
          vector_data.push_back(*tensor.data<int64_t>());
        }
      } else if (data_type == phi::DataType::INT32) {
        const auto& tensor = var->Get<phi::DenseTensor>();
        if (tensor.IsInitialized() &&
            !phi::is_same_place(tensor.place(), expected_place)) {
          phi::DenseTensor tmp_tensor;
          framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
          vector_data.push_back(*tmp_tensor.data<int32_t>());
        } else {
          vector_data.push_back(*tensor.data<int32_t>());
        }
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Data type error. When cast a DenseTensor to VectorTensor, "
            "the data type of DenseTensor must be int32 or int64, "
            "but now data type is %s.",
            data_type));
      }
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported casting input `%s` type to VectorTensor when call pt "
          "kernel.",
          framework::ToTypeName(var->Type())));
    }
  }

  phi::IntArray result{vector_data};
  result.SetFromTensor(true);

  return result;
}

}  // namespace paddle::framework
