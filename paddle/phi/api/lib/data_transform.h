/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace phi {
class DeviceContext;
namespace distributed {
class DistTensor;
class TensorDistAttr;
}  // namespace distributed
}  // namespace phi

namespace paddle {
namespace experimental {

class TransformFlag {
 public:
  TransformFlag(bool stop_transform = false,
                bool trans_dtype = false,
                bool trans_backend = true,
                bool trans_layout = true)
      : stop_transform_(stop_transform),
        trans_data_type_(trans_dtype),
        trans_backend_(trans_backend),
        trans_layout_(trans_layout) {}

  bool NeedTransform() const {
    return !stop_transform_ &&
           (trans_data_type_ || trans_backend_ || trans_layout_);
  }

  bool need_trans_data_type() const {
    return !stop_transform_ && trans_data_type_;
  }

  bool need_trans_backend() const { return !stop_transform_ && trans_backend_; }

  bool need_trans_layout() const { return !stop_transform_ && trans_layout_; }

 private:
  // This is the highest priority in flags,
  // and can be setted by api[data_transform->skip_transform] in the yaml file.
  bool stop_transform_ = false;

  // trans_data_type_ can be setted by api[data_transform->support_trans_dtype]
  // in the yaml file.
  // trans_data_type_ only affect the non complex types,
  // the complex is always transferred, except stop_transform_ is true.
  bool trans_data_type_ = false;

  // trans_backend_ and trans_layout_ are true default,
  // and they can only be setted by global flag.
  bool trans_backend_ = true;
  bool trans_layout_ = true;
};

static inline phi::TensorArgDef GetKernelInputArgDef(
    const phi::TensorArgDef& input_def, phi::Backend kernel_backend) {
  phi::TensorArgDef input_actual_def = input_def;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  // When the backend of input tensor arg_def is CUSTOM, we need to set it to
  // the actual backend by expected_kernel_key.
  if (input_actual_def.backend == phi::Backend::CUSTOM) {
    input_actual_def.SetBackend(kernel_backend);
  }
#endif
  return input_actual_def;
}

std::shared_ptr<phi::DenseTensor> PrepareData(
    const Tensor& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

paddle::optional<phi::DenseTensor> PrepareData(
    const paddle::optional<Tensor>& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

std::unique_ptr<std::vector<phi::DenseTensor>> PrepareData(
    const std::vector<Tensor>& inputs,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

paddle::optional<std::vector<phi::DenseTensor>> PrepareData(
    const paddle::optional<std::vector<Tensor>>& inputs,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

// Only support transferring place for SelectedRows
std::shared_ptr<phi::SelectedRows> PrepareDataForSelectedRows(
    const Tensor& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag);

paddle::optional<phi::SelectedRows> PrepareDataForSelectedRows(
    const paddle::optional<Tensor>& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag);

// Only support transferring contiguous for SparseCooTensor
std::shared_ptr<phi::SparseCooTensor> PrepareDataForSparseCooTensor(
    const Tensor& input);

paddle::optional<phi::SparseCooTensor> PrepareDataForSparseCooTensor(
    const paddle::optional<Tensor>& input);

// Only support transferring contiguous for SparseCsrTensor
std::shared_ptr<phi::SparseCsrTensor> PrepareDataForSparseCsrTensor(
    const Tensor& input);

paddle::optional<phi::SparseCsrTensor> PrepareDataForSparseCsrTensor(
    const paddle::optional<Tensor>& input);

// Only support transferring contiguous
std::shared_ptr<phi::DenseTensor> PrepareDataForDenseTensorInSparse(
    const Tensor& input);

paddle::optional<phi::DenseTensor> PrepareDataForDenseTensorInSparse(
    const paddle::optional<Tensor>& input);

void TransDataBackend(const phi::DenseTensor* tensor,
                      Backend target_backend,
                      phi::DenseTensor* out);

void TransDataBackend(const std::vector<phi::DenseTensor*>& tensor,
                      Backend target_backend,
                      std::vector<phi::DenseTensor*> out);

void TransDataBackend(const phi::SelectedRows* tensor,
                      Backend target_backend,
                      phi::SelectedRows* out);

phi::DenseTensor Trans2Contiguous(const phi::DenseTensor& tensor);

void CheckAndTrans2Contiguous(phi::DenseTensor* tensor);

phi::DenseTensor CheckAndTrans2NewContiguousTensor(
    const phi::DenseTensor& tensor);

std::vector<phi::DenseTensor> CheckAndTrans2NewContiguousTensor(
    const std::vector<phi::DenseTensor>& tensor);

inline bool NeedTransformPlace(const phi::Place& src_place,
                               const Backend& target,
                               const TransformFlag& transform_flag) {
  // NOTE(dev): The default value of TransformFlag is True, if it is set with
  // False
  // somewhere such as ops.yaml or backward.yaml that means we should skip data
  // transform. Because "stop_transform_" has highest priority.
  if (!transform_flag.need_trans_backend()) {
    return false;
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  bool ret = src_place.GetType() == AllocationType::GPUPINNED ||
             (target != Backend::ALL_BACKEND &&
              phi::TransToPhiBackend(src_place) !=
                  (target != Backend::GPUDNN ? target : Backend::GPU));
#elif defined(PADDLE_WITH_XPU)
  bool ret = target != Backend::ALL_BACKEND &&
             phi::TransToPhiBackend(src_place) != target;
#elif defined(PADDLE_WITH_IPU)
  bool ret = target != Backend::ALL_BACKEND &&
             phi::TransToPhiBackend(src_place) != target;
#elif defined(PADDLE_WITH_CUSTOM_DEVICE)
  bool ret = target != Backend::ALL_BACKEND;
  if (target == Backend::CUSTOM) {
    ret = ret && !is_custom_place(src_place);
  } else {
    ret = ret && phi::TransToPhiBackend(src_place) != target;
  }
#else
  bool ret = false;
#endif

#ifdef PADDLE_WITH_DNNL
  if (target == Backend::ONEDNN) {
    ret = src_place.GetType() != AllocationType::CPU;
  }
#endif
  return ret;
}

/* ------------------ for auto parallel ----------------------- */

std::shared_ptr<phi::distributed::DistTensor> ReshardApiInputToKernelInput(
    phi::DeviceContext* dev_ctx,
    const Tensor& tensor,
    const phi::distributed::ArgDistAttr& dist_attr,
    const std::string& arg_name = "");

std::vector<std::shared_ptr<phi::distributed::DistTensor>>
ReshardApiInputToKernelInput(phi::DeviceContext* dev_ctx,
                             const std::vector<Tensor>& tensor,
                             const phi::distributed::ArgDistAttr& dist_attr,
                             const std::string& arg_name = "");

paddle::optional<std::shared_ptr<phi::distributed::DistTensor>>
ReshardApiInputToKernelInput(phi::DeviceContext* dev_ctx,
                             const paddle::optional<Tensor>& tensor,
                             const phi::distributed::ArgDistAttr& dist_attr,
                             const std::string& arg_name = "");

paddle::optional<std::vector<std::shared_ptr<phi::distributed::DistTensor>>>
ReshardApiInputToKernelInput(
    phi::DeviceContext* dev_ctx,
    const paddle::optional<std::vector<Tensor>>& tensors,
    const phi::distributed::ArgDistAttr& dist_attr,
    const std::string& arg_name = "");

void SetInplaceOutputCorrectDistAttr(
    phi::DeviceContext* dev_ctx,
    Tensor& tensor,  // NOLINT
    const phi::distributed::TensorDistAttr& dist_attr,
    bool use_general_spmd_rule = true);

void SetInplaceOutputCorrectDistAttr(
    phi::DeviceContext* dev_ctx,
    Tensor& tensor,  // NOLINT
    const phi::distributed::ArgDistAttr& dist_attr,
    bool use_general_spmd_rule = true);

void SetInplaceOutputCorrectDistAttr(
    phi::DeviceContext* dev_ctx,
    std::vector<Tensor>& tensors,  // NOLINT
    const std::vector<phi::distributed::TensorDistAttr>& dist_attr,
    bool use_general_spmd_rule = true);

void SetInplaceOutputCorrectDistAttr(
    phi::DeviceContext* dev_ctx,
    std::vector<Tensor>& tensors,  // NOLINT
    const phi::distributed::ArgDistAttr& dist_attr,
    bool use_general_spmd_rule = true);

void ReshardOutputPartialAxisToReplicated(
    phi::DeviceContext* dev_ctx, phi::distributed::DistTensor* out_tensor);

void ReshardKernelOutputToApiOutput(
    phi::DeviceContext* dev_ctx,
    const std::shared_ptr<phi::distributed::DistTensor>& src_tensor,
    Tensor* dst_tensor,
    const std::string& arg_name = "");

void ReshardKernelOutputToApiOutput(
    phi::DeviceContext* dev_ctx,
    const std::vector<std::shared_ptr<phi::distributed::DistTensor>>&
        src_tensors,
    const std::vector<Tensor*>& dst_tensors,
    const std::string& arg_name = "");

std::shared_ptr<phi::distributed::DistTensor> PrepareDataForDistTensor(
    std::shared_ptr<phi::distributed::DistTensor> input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

std::vector<std::shared_ptr<phi::distributed::DistTensor>>
PrepareDataForDistTensor(
    std::vector<std::shared_ptr<phi::distributed::DistTensor>> input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

paddle::optional<std::shared_ptr<phi::distributed::DistTensor>>
PrepareDataForDistTensor(
    paddle::optional<std::shared_ptr<phi::distributed::DistTensor>> input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

paddle::optional<std::vector<std::shared_ptr<phi::distributed::DistTensor>>>
PrepareDataForDistTensor(
    paddle::optional<std::vector<std::shared_ptr<phi::distributed::DistTensor>>>
        input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

std::string ReshardDebugInfo(const phi::distributed::DistTensor& src_tensor,
                             const phi::distributed::TensorDistAttr& dist_attr);

}  // namespace experimental
}  // namespace paddle
