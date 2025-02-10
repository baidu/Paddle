//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/kernel_registry.h"

#include <typeindex>
#include <typeinfo>

#include "paddle/phi/core/custom_kernel.h"
#include "paddle/phi/core/kernel_utils.h"
#include "paddle/phi/core/vocab/string_array.h"

namespace phi {

void SetKernelArgsDef(const std::vector<std::type_index>& args_type,
                      const KernelKey& default_key,
                      KernelArgsDef* args_def) {
  auto default_tensor_layout = phi::DataLayout::NCHW;
  if (default_key.layout() != phi::DataLayout::ANY) {
    default_tensor_layout = default_key.layout();
  }
  for (auto arg_type : args_type) {
    if (arg_type == std::type_index(typeid(const CPUContext&))
#if defined(PADDLE_WITH_DNNL)
        || arg_type == std::type_index(typeid(const OneDNNContext&))
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        || arg_type == std::type_index(typeid(const GPUContext&))
#elif defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
        || arg_type == std::type_index(typeid(const XPUContext&))
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_KP)
          || arg_type == std::type_index(typeid(const KPSContext&))
#endif
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
        || arg_type == std::type_index(typeid(const CustomContext&))) {
#else
    ) {
#endif
      // do nothing, skip context arg now
    } else if (arg_type ==
               std::type_index(typeid(const DenseTensor&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(
                   typeid(const paddle::optional<DenseTensor>&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(
                   typeid(const paddle::optional<
                          std::vector<const DenseTensor*>>&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(
                   typeid(const paddle::optional<SelectedRows>&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(
                   typeid(const std::vector<const DenseTensor*>&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(const phi::ExtendedTensor&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(
                   const paddle::optional<ExtendedTensor>&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(
                   const std::vector<const ExtendedTensor*>&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(const phi::FeedList&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type == std::type_index(typeid(
                               const paddle::optional<Strings>&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(
                   const std::vector<const SelectedRows*>&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(
                   typeid(const std::vector<const TensorBase*>&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(
                   typeid(const std::vector<const TensorArray*>&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(const SelectedRows&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(const StringTensor&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(const SparseCooTensor&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(
                   paddle::optional<const SparseCooTensor&>))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(const SparseCsrTensor&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(
                   paddle::optional<const SparseCsrTensor&>))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type ==
               std::type_index(typeid(const TensorArray&))) {  // NOLINT
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            arg_type);
    } else if (arg_type == std::type_index(typeid(DenseTensor*))) {  // NOLINT
      args_def->AppendOutput(default_key.backend(),
                             default_tensor_layout,
                             default_key.dtype(),
                             arg_type);
    } else if (arg_type ==
               std::type_index(typeid(std::vector<DenseTensor*>))) {  // NOLINT
      args_def->AppendOutput(default_key.backend(),
                             default_tensor_layout,
                             default_key.dtype(),
                             arg_type);
    } else if (arg_type == std::type_index(typeid(SelectedRows*))) {  // NOLINT
      args_def->AppendOutput(default_key.backend(),
                             default_tensor_layout,
                             default_key.dtype(),
                             arg_type);
    } else if (arg_type == std::type_index(typeid(TensorArray*))) {  // NOLINT
      args_def->AppendOutput(default_key.backend(),
                             default_tensor_layout,
                             default_key.dtype(),
                             arg_type);
    } else if (arg_type ==
               std::type_index(typeid(SparseCooTensor*))) {  // NOLINT
      args_def->AppendOutput(default_key.backend(),
                             default_tensor_layout,
                             default_key.dtype(),
                             arg_type);
    } else if (arg_type ==
               std::type_index(typeid(SparseCsrTensor*))) {  // NOLINT
      args_def->AppendOutput(default_key.backend(),
                             default_tensor_layout,
                             default_key.dtype(),
                             arg_type);
    } else if (arg_type == std::type_index(typeid(StringTensor*))) {  // NOLINT
      args_def->AppendOutput(default_key.backend(),
                             default_tensor_layout,
                             default_key.dtype(),
                             arg_type);
    } else if (arg_type == std::type_index(typeid(ExtendedTensor*)) ||
               arg_type ==
                   std::type_index(typeid(std::vector<ExtendedTensor*>)) ||
               arg_type == std::type_index(typeid(std::vector<Vocab*>)) ||
               arg_type ==
                   std::type_index(typeid(phi::FetchList*))) {  // NOLINT
      args_def->AppendOutput(default_key.backend(),
                             default_tensor_layout,
                             default_key.dtype(),
                             arg_type);
    } else if (arg_type == std::type_index(typeid(bool))) {
      args_def->AppendAttribute(AttributeType::BOOL);
    } else if (arg_type == std::type_index(typeid(int))) {
      args_def->AppendAttribute(AttributeType::INT32);
    } else if (arg_type == std::type_index(typeid(int64_t))) {
      args_def->AppendAttribute(AttributeType::INT64);
    } else if (arg_type == std::type_index(typeid(float))) {
      args_def->AppendAttribute(AttributeType::FLOAT32);
    } else if (arg_type == std::type_index(typeid(double))) {
      args_def->AppendAttribute(AttributeType::FLOAT64);
    } else if (arg_type == std::type_index(typeid(std::string))) {
      args_def->AppendAttribute(AttributeType::STRING);
    } else if (arg_type == std::type_index(typeid(const std::vector<bool>&))) {
      args_def->AppendAttribute(AttributeType::BOOLS);
    } else if (arg_type == std::type_index(typeid(const std::vector<int>&))) {
      args_def->AppendAttribute(AttributeType::INT32S);
    } else if (arg_type ==
               std::type_index(typeid(const std::vector<int64_t>&))) {
      args_def->AppendAttribute(AttributeType::INT64S);
    } else if (arg_type == std::type_index(typeid(const std::vector<float>&))) {
      args_def->AppendAttribute(AttributeType::FLOAT32S);
    } else if (arg_type ==
               std::type_index(typeid(const std::vector<double>&))) {
      args_def->AppendAttribute(AttributeType::FLOAT64S);
    } else if (arg_type ==
               std::type_index(typeid(const std::vector<std::string>&))) {
      args_def->AppendAttribute(AttributeType::STRINGS);
    } else if (arg_type == std::type_index(typeid(const Scalar&))) {
      args_def->AppendAttribute(AttributeType::SCALAR);
    } else if (arg_type ==
               std::type_index(typeid(const std::vector<Scalar>&))) {
      args_def->AppendAttribute(AttributeType::SCALARS);
    } else if (arg_type == std::type_index(typeid(const IntArray&))) {
      args_def->AppendAttribute(AttributeType::INT_ARRAY);
    } else if (arg_type == std::type_index(typeid(DataType))) {
      args_def->AppendAttribute(AttributeType::DATA_TYPE);
    } else if (arg_type == std::type_index(typeid(DataLayout))) {
      args_def->AppendAttribute(AttributeType::DATA_LAYOUT);
    } else if (arg_type == std::type_index(typeid(Place))) {
      args_def->AppendAttribute(AttributeType::PLACE);
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "Unsupported kernel argument type `%s`.", arg_type.name()));
    }
  }
}
}  // namespace phi
