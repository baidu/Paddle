// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/grad_node_info.h"

#include "glog/logging.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

/**
 * Implementation of GradNodeBase, Edge and GradTensorHolder.
 **/
namespace egr {

static void CheckTensor(const paddle::Tensor& pre, const paddle::Tensor& post) {
  if (!pre.has_allocation() && post.has_allocation()) {
    PADDLE_THROW(common::errors::PermissionDenied(
        "The tensor in before and after hook are not consistent"));
  }
  if (pre.has_allocation() && post.has_allocation()) {
    VLOG(7) << phi::DataTypeToString(pre.dtype()) << " "
            << phi::DataTypeToString(post.dtype());
    PADDLE_ENFORCE_EQ(
        pre.dtype(),
        post.dtype(),
        common::errors::PermissionDenied(
            "The dtype of tensor before(%s) and after(%s) hook are not "
            "consistent",
            phi::DataTypeToString(pre.dtype()),
            phi::DataTypeToString(post.dtype())));
    PADDLE_ENFORCE_EQ(pre.place(),
                      post.place(),
                      common::errors::PermissionDenied(
                          "The place of tensor before(%s) and after(%s) "
                          "hook are not consistent",
                          pre.place().DebugString(),
                          post.place().DebugString()));
  }
}

GradNodeBase::GradNodeBase(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
    : bwd_out_meta_(), bwd_in_meta_(), gradient_hooks_() {
  VLOG(7) << "Construct GradNodeBase";
  bwd_in_meta_.resize(bwd_in_slot_num);
  bwd_out_meta_.resize(bwd_out_slot_num);
}

const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
GradNodeBase::InputMeta() const {
  return bwd_in_meta_;
}

const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
GradNodeBase::OutputMeta() const {
  return bwd_out_meta_;
}

paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
GradNodeBase::MutableOutputMeta() {
  return bwd_out_meta_;
}

void GradNodeBase::SetGradInMeta(const paddle::Tensor& fwd_out,
                                 size_t slot_rank) {
  VLOG(7) << "Set GradSlotMeta for Grad Inputs";
  auto* fwd_out_meta = egr::EagerUtils::nullable_autograd_meta(fwd_out);
  PADDLE_ENFORCE_LE(
      slot_rank,
      (bwd_in_meta_.size() - 1),
      common::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_in_meta_ size, since "
          "bwd_in_meta_ is designed to hold as same num as backward "
          "inputs."));
  auto& metas = bwd_in_meta_.at(slot_rank);
  if (metas.empty()) {
    metas.resize(1);
  }

  auto& meta = metas[0];
  if (fwd_out_meta && fwd_out_meta->StopGradient()) {
    meta.SetStopGradient(fwd_out_meta->StopGradient());
  }

  if (!fwd_out.has_allocation()) {
    if (fwd_out.defined() && fwd_out.is_dist_tensor() &&
        phi::distributed::NeedComputationClipForPP(fwd_out.impl())) {
      VLOG(3) << "Tensor " << fwd_out.name() << " is DistTensor,"
              << " and needs computation clip for pipeline parallel."
              << " Still SetGradInMeta for it.";
    } else {
      VLOG(7)
          << "Skip Configuring GradSlotMeta for uninitialized GradInput Tensor";
      return;
    }
  }

  const phi::DenseTensor* dense_tensor = nullptr;
  // Record TensorMeta
  if (phi::DenseTensor::classof(fwd_out.impl().get())) {
    // Only Copy Meta
    dense_tensor = static_cast<phi::DenseTensor*>(fwd_out.impl().get());
  } else if (phi::SparseCooTensor::classof(fwd_out.impl().get())) {
    phi::SparseCooTensor* coo_tensor =
        static_cast<phi::SparseCooTensor*>(fwd_out.impl().get());
    dense_tensor = coo_tensor->mutable_non_zero_elements();
  } else if (phi::SparseCsrTensor::classof(fwd_out.impl().get())) {
    phi::SparseCsrTensor* csr_tensor =
        static_cast<phi::SparseCsrTensor*>(fwd_out.impl().get());
    dense_tensor = csr_tensor->mutable_non_zero_elements();
  } else if (phi::distributed::DistTensor::classof(fwd_out.impl().get())) {
    dense_tensor =  // NOLINT
        &(static_cast<phi::distributed::DistTensor*>(fwd_out.impl().get())
              ->value());
    meta.SetDistAttr(
        static_cast<phi::distributed::DistTensor*>(fwd_out.impl().get())
            ->dist_attr());
    meta.SetDistTensorGlobalDims(
        static_cast<phi::distributed::DistTensor*>(fwd_out.impl().get())
            ->dims());
    SetIsRunAutoParallel(true);
  } else {
    VLOG(7) << "Unable to initialize the DenseTensorMeta of GradSlotMeta with "
               "non-DenseTensor argument.";
  }
  PADDLE_ENFORCE_NE(
      dense_tensor->meta().dtype,
      phi::DataType::UNDEFINED,
      common::errors::Fatal(
          "Attempting to copy DenseTensorMeta with phi::DataType::UNDEFINED,"
          "which is illegal."));

  meta.SetTensorMeta(dense_tensor->meta());
  meta.SetPlace(fwd_out.place());

  if (dense_tensor->type() == phi::DataType::COMPLEX64 ||
      dense_tensor->type() == phi::DataType::COMPLEX128) {
    need_complex_to_real_ = true;
  }
}

void GradNodeBase::SetGradInMeta(const std::vector<paddle::Tensor>& fwd_out,
                                 size_t slot_rank) {
  VLOG(7) << "Set GradSlotMeta for Grad Inputs";
  size_t slot_size = fwd_out.size();
  PADDLE_ENFORCE_LE(
      slot_rank,
      (bwd_in_meta_.size() - 1),
      common::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_in_meta_ size, since "
          "bwd_in_meta_ is designed to hold as same num as backward "
          "inputs."));
  auto& metas = bwd_in_meta_.at(slot_rank);
  // Init stop gradient vector before use to avoid push back
  if (metas.size() < slot_size) {
    VLOG(7) << "Init bwd_in_meta_ with slot rank: " << slot_rank;
    metas.resize(slot_size);
  }
  for (size_t i = 0; i < slot_size; i++) {
    auto& meta = metas[i];
    const auto& fwd_out_tensor = fwd_out[i];
    auto* fwd_out_meta =
        egr::EagerUtils::nullable_autograd_meta(fwd_out_tensor);
    PADDLE_ENFORCE_NOT_NULL(fwd_out_meta,
                            common::errors::PreconditionNotMet(
                                "Bwd_in_meta should only be called while "
                                "autograd_meta is not null. If you got this "
                                "error, it indicates bugs in framework."));
    if (fwd_out_meta && fwd_out_meta->StopGradient()) {
      // Set Stop Gradient only when its true or non-initialized autograd_meta,
      // since all default value is false.
      meta.SetStopGradient(fwd_out_meta->StopGradient());
    }

    if (!fwd_out_tensor.has_allocation()) {
      if (fwd_out_tensor.defined() && fwd_out_tensor.is_dist_tensor() &&
          phi::distributed::NeedComputationClipForPP(fwd_out_tensor.impl())) {
        VLOG(3) << "Tensor " << fwd_out_tensor.name() << " is DistTensor,"
                << " and needs computation clip for pipeline parallel."
                << " Still SetGradInMeta for it.";
      } else {
        VLOG(7) << "Skip Configuring GradSlotMeta for uninitialized GradInput "
                   "Tensor";
        return;
      }
    }

    // Record TensorMeta
    if (phi::DenseTensor::classof(fwd_out_tensor.impl().get())) {
      // Only Copy Meta
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(fwd_out_tensor.impl().get());

      PADDLE_ENFORCE_NE(
          dense_tensor->meta().dtype,
          phi::DataType::UNDEFINED,
          common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                "with phi::DataType::UNDEFINED,"
                                "which is illegal."));
      meta.SetTensorMeta(dense_tensor->meta());
      meta.SetPlace(fwd_out_tensor.place());

      if (dense_tensor->type() == phi::DataType::COMPLEX64 ||
          dense_tensor->type() == phi::DataType::COMPLEX128) {
        need_complex_to_real_ = true;
      }
    } else if (phi::distributed::DistTensor::classof(
                   fwd_out_tensor.impl().get())) {
      // Only Copy Meta
      meta.SetDistAttr(static_cast<phi::distributed::DistTensor*>(
                           fwd_out_tensor.impl().get())
                           ->dist_attr());
      meta.SetDistTensorGlobalDims(static_cast<phi::distributed::DistTensor*>(
                                       fwd_out_tensor.impl().get())
                                       ->dims());
      SetIsRunAutoParallel(true);

      auto dense_tensor = static_cast<phi::distributed::DistTensor*>(
                              fwd_out_tensor.impl().get())
                              ->value();

      PADDLE_ENFORCE_NE(
          dense_tensor.meta().dtype,
          phi::DataType::UNDEFINED,
          common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                "with phi::DataType::UNDEFINED,"
                                "which is illegal."));
      meta.SetTensorMeta(dense_tensor.meta());
      meta.SetPlace(fwd_out_tensor.place());

      if (dense_tensor.type() == phi::DataType::COMPLEX64 ||
          dense_tensor.type() == phi::DataType::COMPLEX128) {
        need_complex_to_real_ = true;
      }
    } else {
      VLOG(7) << "Unable to initialize the DenseTensorMeta of GradSlotMeta "
                 "with non-DenseTensor argument.";
    }
  }
}

void GradNodeBase::SetGradInMeta(const std::vector<paddle::Tensor*>& fwd_out,
                                 size_t slot_rank) {
  VLOG(7) << "Set GradSlotMeta for Grad Inputs";
  size_t slot_size = fwd_out.size();
  PADDLE_ENFORCE_LE(
      slot_rank,
      (bwd_in_meta_.size() - 1),
      common::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_in_meta_ size, since "
          "bwd_in_meta_ is designed to hold as same num as backward "
          "inputs."));
  auto& metas = bwd_in_meta_.at(slot_rank);
  // Init stop gradient vector before use to avoid push back
  if (metas.size() < slot_size) {
    VLOG(7) << "Init bwd_in_meta_ with slot rank: " << slot_rank;
    metas.resize(slot_size);
  }
  for (size_t i = 0; i < slot_size; i++) {
    auto& meta = metas[i];
    const auto& fwd_out_tensor = *fwd_out[i];
    auto* fwd_out_meta =
        egr::EagerUtils::nullable_autograd_meta(fwd_out_tensor);
    PADDLE_ENFORCE_NOT_NULL(fwd_out_meta,
                            common::errors::PreconditionNotMet(
                                "Bwd_in_meta should only be called while "
                                "autograd_meta is not null. If you got this "
                                "error, it indicates bugs in framework."));
    if (fwd_out_meta && fwd_out_meta->StopGradient()) {
      // Set Stop Gradient only when its true or non-initialized autograd_meta,
      // since all default value is false.
      meta.SetStopGradient(fwd_out_meta->StopGradient());
    }

    if (!fwd_out_tensor.has_allocation()) {
      if (fwd_out_tensor.defined() && fwd_out_tensor.is_dist_tensor() &&
          phi::distributed::NeedComputationClipForPP(fwd_out_tensor.impl())) {
        VLOG(3) << "Tensor " << fwd_out_tensor.name() << " is DistTensor,"
                << " and needs computation clip for pipeline parallel."
                << " Still SetGradInMeta for it.";
      } else {
        VLOG(7) << "Skip Configuring GradSlotMeta for uninitialized GradInput "
                   "Tensor";
        return;
      }
    }

    // Record TensorMeta
    if (phi::DenseTensor::classof(fwd_out_tensor.impl().get())) {
      // Only Copy Meta
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(fwd_out_tensor.impl().get());

      PADDLE_ENFORCE_NE(
          dense_tensor->meta().dtype,
          phi::DataType::UNDEFINED,
          common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                "with phi::DataType::UNDEFINED,"
                                "which is illegal."));
      meta.SetTensorMeta(dense_tensor->meta());
      meta.SetPlace(fwd_out_tensor.place());

      if (dense_tensor->type() == phi::DataType::COMPLEX64 ||
          dense_tensor->type() == phi::DataType::COMPLEX128) {
        need_complex_to_real_ = true;
      }
    } else if (phi::distributed::DistTensor::classof(
                   fwd_out_tensor.impl().get())) {
      // Only Copy Meta
      meta.SetDistAttr(static_cast<phi::distributed::DistTensor*>(
                           fwd_out_tensor.impl().get())
                           ->dist_attr());
      meta.SetDistTensorGlobalDims(static_cast<phi::distributed::DistTensor*>(
                                       fwd_out_tensor.impl().get())
                                       ->dims());
      SetIsRunAutoParallel(true);

      auto dense_tensor = static_cast<phi::distributed::DistTensor*>(
                              fwd_out_tensor.impl().get())
                              ->value();

      PADDLE_ENFORCE_NE(
          dense_tensor.meta().dtype,
          phi::DataType::UNDEFINED,
          common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                "with phi::DataType::UNDEFINED,"
                                "which is illegal."));
      meta.SetTensorMeta(dense_tensor.meta());
      meta.SetPlace(fwd_out_tensor.place());

      if (dense_tensor.type() == phi::DataType::COMPLEX64 ||
          dense_tensor.type() == phi::DataType::COMPLEX128) {
        need_complex_to_real_ = true;
      }
    } else {
      VLOG(7) << "Unable to initialize the DenseTensorMeta of GradSlotMeta "
                 "with non-DenseTensor argument.";
    }
  }
}

void GradNodeBase::SetGradOutMeta(const paddle::Tensor& fwd_in,
                                  size_t slot_rank) {
  auto* fwd_in_meta = egr::EagerUtils::nullable_autograd_meta(fwd_in);
  PADDLE_ENFORCE_LE(
      (slot_rank + 1),
      bwd_out_meta_.size(),
      common::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_out_meta_ size, "
          "since bwd_out_meta_ is designed to hold as same num as "
          "backward outputs."));
  auto& metas = bwd_out_meta_.at(slot_rank);
  // Init stop gradient vector before use to avoid push back
  if (metas.empty()) {
    metas.resize(1);
  }
  auto& meta = metas[0];
  // Set Stop_gradient
  if (fwd_in_meta) {
    meta.SetStopGradient(fwd_in_meta->StopGradient());
  } else {
    meta.SetStopGradient(true);
  }
  // Set Adj Edges
  if (fwd_in_meta && !fwd_in_meta->StopGradient()) {
    auto node = fwd_in_meta->GetMutableGradNode();
    if (!node || !node.get()) {
      fwd_in_meta->SetGradNode(
          std::make_shared<egr::GradNodeAccumulation>(fwd_in_meta));
    }
    VLOG(3) << "Add Edges for slot: " << slot_rank << ", the Edge is from "
            << this->name() << " (addr: " << this << ") "
            << " to " << fwd_in_meta->GetMutableGradNode()->name()
            << " (addr: " << fwd_in_meta->GetMutableGradNode().get() << ")";

    meta.SetEdge(fwd_in_meta->GetMutableGradNode(), fwd_in_meta->OutRankInfo());
  }
  // Record TensorMeta
  if (fwd_in.impl() && fwd_in.impl().get()) {
    if (phi::DenseTensor::classof(fwd_in.impl().get())) {
      // Only Copy Meta
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(fwd_in.impl().get());
      PADDLE_ENFORCE_NE(
          dense_tensor->meta().dtype,
          phi::DataType::UNDEFINED,
          common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                "with phi::DataType::UNDEFINED,"
                                "which is illegal."));
      meta.SetTensorMeta(dense_tensor->meta());
      meta.SetPlace(fwd_in.place());
    } else if (phi::distributed::DistTensor::classof(fwd_in.impl().get())) {
      const phi::distributed::DistTensor* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(fwd_in.impl().get());
      const phi::DenseTensor& dense_tensor = dist_tensor->value();
      PADDLE_ENFORCE_NE(
          dense_tensor.meta().dtype,
          phi::DataType::UNDEFINED,
          common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                "with phi::DataType::UNDEFINED,"
                                "which is illegal."));
      meta.SetTensorMeta(dense_tensor.meta());
      meta.SetPlace(fwd_in.place());
      // Set DistAttr
      // Forward input DistTensor could be uninitialized.
      PADDLE_ENFORCE_NE(
          dist_tensor->dist_attr().empty(),
          true,
          common::errors::InvalidArgument(
              "The forward input DistTensor's dist attr is empty."));
      auto dist_attr = dist_tensor->dist_attr();
      dist_attr.clean_partial_status();
      meta.SetDistAttr(dist_attr);
      meta.SetDistTensorGlobalDims(dist_tensor->dims());
      SetIsRunAutoParallel(true);
    } else if (phi::SparseCsrTensor::classof(fwd_in.impl().get())) {
      phi::SparseCsrTensor* sparse_tensor =
          static_cast<phi::SparseCsrTensor*>(fwd_in.impl().get());
      const phi::DenseTensor dense_tensor =
          static_cast<const phi::DenseTensor>(sparse_tensor->values());
      PADDLE_ENFORCE_NE(
          dense_tensor.dtype(),
          phi::DataType::UNDEFINED,
          common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                "with phi::DataType::UNDEFINED,"
                                "which is illegal."));
      meta.SetTensorMeta(dense_tensor.meta());
      meta.SetPlace(fwd_in.place());
    } else if (phi::SparseCooTensor::classof(fwd_in.impl().get())) {
      phi::SparseCooTensor* sparse_tensor =
          static_cast<phi::SparseCooTensor*>(fwd_in.impl().get());
      const phi::DenseTensor dense_tensor =
          static_cast<const phi::DenseTensor>(sparse_tensor->values());
      PADDLE_ENFORCE_NE(
          dense_tensor.dtype(),
          phi::DataType::UNDEFINED,
          common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                "with phi::DataType::UNDEFINED,"
                                "which is illegal."));
      meta.SetTensorMeta(dense_tensor.meta());
      meta.SetPlace(fwd_in.place());
    } else {
      VLOG(7)
          << "Unable to initialize the DenseTensorMeta of GradSlotMeta with "
             "non-DenseTensor argument.";
    }
  } else {
    VLOG(7) << "Unable to initialize the DenseTensorMeta because the Tensor "
               "is not initialized.";
  }
}

/*
special func for matmul_double_grad etc. dx exists when x and y exists,
if stop_gradient of y is true, dy is None who is matmul_grad's out_put, ddy is
None, so dx = ddy * dout should be None.
*/
void GradNodeBase::SetGradOutMeta(const paddle::Tensor& fwd_in,
                                  const AutogradMeta* fwd_out_meta,
                                  size_t slot_rank) {
  auto* fwd_in_meta = egr::EagerUtils::nullable_autograd_meta(fwd_in);
  PADDLE_ENFORCE_LE(
      (slot_rank + 1),
      bwd_out_meta_.size(),
      common::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_out_meta_ size, "
          "since bwd_out_meta_ is designed to hold as same num as "
          "backward outputs."));
  auto& metas = bwd_out_meta_.at(slot_rank);
  // Init stop gradient vector before use to avoid push back
  if (metas.empty()) {
    metas.resize(1);
  }
  auto& meta = metas[0];
  // Set Stop_gradient
  if (fwd_in_meta && !fwd_in_meta->StopGradient() && fwd_out_meta) {
    meta.SetStopGradient(false);
  } else {
    meta.SetStopGradient(true);
  }
  // Set Adj Edges
  if (fwd_in_meta && !fwd_in_meta->StopGradient() && fwd_out_meta) {
    auto node = fwd_in_meta->GetMutableGradNode();
    if (!node || !node.get()) {
      fwd_in_meta->SetGradNode(
          std::make_shared<egr::GradNodeAccumulation>(fwd_in_meta));
    }
    VLOG(3) << "Add Edges for slot: " << slot_rank << ", the Edge is from "
            << this->name() << " (addr: " << this << ") "
            << " to " << fwd_in_meta->GetMutableGradNode()->name()
            << " (addr: " << fwd_in_meta->GetMutableGradNode().get() << ")";

    meta.SetEdge(fwd_in_meta->GetMutableGradNode(), fwd_in_meta->OutRankInfo());
  }
  // Record TensorMeta
  if (fwd_in.impl() && fwd_in.impl().get()) {
    if (phi::DenseTensor::classof(fwd_in.impl().get())) {
      // Only Copy Meta
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(fwd_in.impl().get());
      PADDLE_ENFORCE_NE(
          dense_tensor->meta().dtype,
          phi::DataType::UNDEFINED,
          common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                "with phi::DataType::UNDEFINED,"
                                "which is illegal."));
      meta.SetTensorMeta(dense_tensor->meta());
      meta.SetPlace(fwd_in.place());
    } else if (phi::distributed::DistTensor::classof(fwd_in.impl().get())) {
      // Only Copy Meta
      meta.SetDistAttr(
          static_cast<phi::distributed::DistTensor*>(fwd_in.impl().get())
              ->dist_attr());
      meta.SetDistTensorGlobalDims(
          static_cast<phi::distributed::DistTensor*>(fwd_in.impl().get())
              ->dims());
      SetIsRunAutoParallel(true);
      auto dense_tensor =
          static_cast<phi::distributed::DistTensor*>(fwd_in.impl().get())
              ->value();
      PADDLE_ENFORCE_NE(
          dense_tensor.meta().dtype,
          phi::DataType::UNDEFINED,
          common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                "with phi::DataType::UNDEFINED,"
                                "which is illegal."));
      meta.SetTensorMeta(dense_tensor.meta());
      meta.SetPlace(fwd_in.place());
    }
  } else {
    VLOG(7) << "Unable to initialize the DenseTensorMeta of GradSlotMeta with "
               "non-DenseTensor argument.";
  }
}

void GradNodeBase::SetGradOutMeta(const std::vector<paddle::Tensor>& fwd_in,
                                  size_t slot_rank) {
  size_t slot_size = fwd_in.size();
  PADDLE_ENFORCE_LE(
      slot_rank,
      (bwd_out_meta_.size() - 1),
      common::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_out_meta_ size, "
          "since bwd_out_meta_ is designed to hold as same num as "
          "backward outputs."));
  auto& metas = bwd_out_meta_.at(slot_rank);
  // Init stop gradient vector before use to avoid push back
  if (metas.size() < slot_size) {
    metas.resize(slot_size);
  }
  for (size_t i = 0; i < slot_size; i++) {
    const auto& fwd_in_tensor = fwd_in[i];
    auto& meta = metas[i];
    auto* fwd_in_meta = egr::EagerUtils::nullable_autograd_meta(fwd_in_tensor);
    // Set Stop_gradient
    if (fwd_in_meta) {
      meta.SetStopGradient(fwd_in_meta->StopGradient());
    }
    // Set Adj Edges
    if (fwd_in_meta && !fwd_in_meta->StopGradient()) {
      auto node = fwd_in_meta->GetMutableGradNode();
      if (!node || !node.get()) {
        fwd_in_meta->SetGradNode(
            std::make_shared<egr::GradNodeAccumulation>(fwd_in_meta));
      }
      VLOG(3) << "Add Edges for slot: " << slot_rank << ", the Edge is from "
              << this->name() << " (addr: " << this << ") "
              << " to " << fwd_in_meta->GetMutableGradNode()->name()
              << " (addr: " << fwd_in_meta->GetMutableGradNode().get() << ")";

      meta.SetEdge(fwd_in_meta->GetMutableGradNode(),
                   fwd_in_meta->OutRankInfo());
    }
    // Record TensorMeta
    if (fwd_in_tensor.impl() && fwd_in_tensor.impl().get()) {
      if (phi::DenseTensor::classof(fwd_in_tensor.impl().get())) {
        phi::DenseTensor* dense_tensor =
            static_cast<phi::DenseTensor*>(fwd_in_tensor.impl().get());
        PADDLE_ENFORCE_NE(
            dense_tensor->dtype(),
            phi::DataType::UNDEFINED,
            common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                  "with phi::DataType::UNDEFINED,"
                                  "which is illegal."));
        meta.SetTensorMeta(dense_tensor->meta());
        meta.SetPlace(fwd_in_tensor.place());
      } else if (phi::distributed::DistTensor::classof(
                     fwd_in_tensor.impl().get())) {
        meta.SetDistAttr(static_cast<phi::distributed::DistTensor*>(
                             fwd_in_tensor.impl().get())
                             ->dist_attr());
        meta.SetDistTensorGlobalDims(static_cast<phi::distributed::DistTensor*>(
                                         fwd_in_tensor.impl().get())
                                         ->dims());
        SetIsRunAutoParallel(true);
        auto dense_tensor = static_cast<phi::distributed::DistTensor*>(
                                fwd_in_tensor.impl().get())
                                ->value();
        PADDLE_ENFORCE_NE(
            dense_tensor.dtype(),
            phi::DataType::UNDEFINED,
            common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                  "with phi::DataType::UNDEFINED,"
                                  "which is illegal."));
        meta.SetTensorMeta(dense_tensor.meta());
        meta.SetPlace(fwd_in_tensor.place());
      }
    } else {
      VLOG(7)
          << "Unable to initialize the DenseTensorMeta of GradSlotMeta with "
             "non-DenseTensor argument.";
    }
  }
}

void GradNodeBase::SetGradOutMeta(
    const std::vector<const paddle::Tensor*>& fwd_in, size_t slot_rank) {
  size_t slot_size = fwd_in.size();
  PADDLE_ENFORCE_LE(
      slot_rank,
      (bwd_out_meta_.size() - 1),
      common::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_out_meta_ size, "
          "since bwd_out_meta_ is designed to hold as same num as "
          "backward outputs."));
  auto& metas = bwd_out_meta_.at(slot_rank);
  // Init stop gradient vector before use to avoid push back
  if (metas.size() < slot_size) {
    metas.resize(slot_size);
  }
  for (size_t i = 0; i < slot_size; i++) {
    const auto& fwd_in_tensor = (*fwd_in[i]);
    auto& meta = metas[i];
    auto* fwd_in_meta = egr::EagerUtils::nullable_autograd_meta(fwd_in_tensor);
    // Set Stop_gradient
    if (fwd_in_meta) {
      meta.SetStopGradient(fwd_in_meta->StopGradient());
    }
    // Set Adj Edges
    if (fwd_in_meta && !fwd_in_meta->StopGradient()) {
      auto node = fwd_in_meta->GetMutableGradNode();
      if (!node || !node.get()) {
        fwd_in_meta->SetGradNode(
            std::make_shared<egr::GradNodeAccumulation>(fwd_in_meta));
      }
      VLOG(3) << "Add Edges for slot: " << slot_rank << ", the Edge is from "
              << this->name() << " (addr: " << this << ") "
              << " to " << fwd_in_meta->GetMutableGradNode()->name()
              << " (addr: " << fwd_in_meta->GetMutableGradNode().get() << ")";

      meta.SetEdge(fwd_in_meta->GetMutableGradNode(),
                   fwd_in_meta->OutRankInfo());
    }
    // Record TensorMeta
    if (fwd_in_tensor.impl() && fwd_in_tensor.impl().get()) {
      if (phi::DenseTensor::classof(fwd_in_tensor.impl().get())) {
        // Only Copy Meta
        phi::DenseTensor* dense_tensor =
            static_cast<phi::DenseTensor*>(fwd_in_tensor.impl().get());
        PADDLE_ENFORCE_NE(
            dense_tensor->dtype(),
            phi::DataType::UNDEFINED,
            common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                  "with phi::DataType::UNDEFINED,"
                                  "which is illegal."));
        meta.SetTensorMeta(dense_tensor->meta());
        meta.SetPlace(fwd_in_tensor.place());
      } else if (phi::distributed::DistTensor::classof(
                     fwd_in_tensor.impl().get())) {
        // Only Copy Meta
        meta.SetDistAttr(static_cast<phi::distributed::DistTensor*>(
                             fwd_in_tensor.impl().get())
                             ->dist_attr());
        meta.SetDistTensorGlobalDims(static_cast<phi::distributed::DistTensor*>(
                                         fwd_in_tensor.impl().get())
                                         ->dims());
        SetIsRunAutoParallel(true);
        auto dense_tensor = static_cast<phi::distributed::DistTensor*>(
                                fwd_in_tensor.impl().get())
                                ->value();
        PADDLE_ENFORCE_NE(
            dense_tensor.dtype(),
            phi::DataType::UNDEFINED,
            common::errors::Fatal("Attempting to copy DenseTensorMeta "
                                  "with phi::DataType::UNDEFINED,"
                                  "which is illegal."));
        meta.SetTensorMeta(dense_tensor.meta());
        meta.SetPlace(fwd_in_tensor.place());
      }
    } else {
      VLOG(7)
          << "Unable to initialize the DenseTensorMeta of GradSlotMeta with "
             "non-DenseTensor argument.";
    }
  }
}

void GradNodeBase::SetDefaultGradInOutMeta() {
  PADDLE_ENFORCE((bwd_out_meta_.size() == 1) && (bwd_in_meta_.size() == 1),
                 common::errors::PreconditionNotMet(
                     "We can only support 1 input and 1 output in default grad "
                     "meta setter, other size of inputs and outputs should "
                     "create with Setter and Getters"));
  // Default stop_gradient is false and slot id is 0, slot size is 1;
  bwd_out_meta_[0].resize(1);
  bwd_in_meta_[0].resize(1);
}

int64_t GradNodeBase::RegisterGradientHook(
    size_t slot_id, size_t rank, std::shared_ptr<egr::TensorHook>&& hook) {
  gradient_hooks_.emplace(next_hook_id_,
                          std::make_tuple(slot_id, rank, std::move(hook)));
  return next_hook_id_++;
}

paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
GradNodeBase::ApplyGradientHooks(
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors) {
  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize> outs(
      tensors.size());
  for (auto& hook_pair : gradient_hooks_) {
    size_t slot_id = std::get<0>(hook_pair.second);
    size_t rank = std::get<1>(hook_pair.second);

    auto hook = std::get<2>(hook_pair.second);

    PADDLE_ENFORCE(slot_id < tensors.size(),
                   common::errors::Fatal(
                       "Slot_id from registered hook should be smaller than "
                       "slot size of grad_tensors"));

    PADDLE_ENFORCE(rank < tensors[slot_id].size(),
                   common::errors::Fatal(
                       "rank of slot %d from registered hook should be smaller "
                       "than rank size of grad_tensors",
                       slot_id));

    std::vector<paddle::Tensor>& slot_out = outs[slot_id];
    slot_out.resize(tensors[slot_id].size());
    paddle::Tensor& out = slot_out[rank];
    if (!out.defined() || !out.has_allocation()) {
      out = (*hook)(tensors[slot_id][rank]);
    } else {
      // If more than one hook is registered, the input to the next hook func
      // should be the output of the previous hook
      out = (*hook)(out);
    }
  }

  for (size_t i = 0; i < outs.size(); i++) {
    if (outs[i].empty() && (!tensors[i].empty())) {
      outs[i].resize(tensors[i].size());
    }
    // TODO(Jiabin): Optimize this if we only add hook slot by slot
    for (size_t j = 0; j < outs[i].size(); j++) {
      if (!outs[i][j].defined() || !outs[i][j].initialized()) {
        outs[i][j] = tensors[i][j];
      }
      CheckTensor(tensors[i][j], outs[i][j]);
    }
  }

  return outs;
}

void GradNodeBase::HandleComplexGradToRealGrad(
    paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>*
        out_grads) {
  for (size_t slot_id = 0; slot_id < out_grads->size(); slot_id++) {
    const std::vector<paddle::Tensor>& slot_out_grads = (*out_grads)[slot_id];
    for (size_t rank_id = 0; rank_id < slot_out_grads.size(); rank_id++) {
      if (bwd_out_meta_[slot_id].size() == 0) continue;
      const GradSlotMeta& slot_meta = bwd_out_meta_[slot_id][rank_id];
      PADDLE_ENFORCE(
          slot_meta.HasTensorMeta() > 0,
          common::errors::Fatal(
              "We require TensorMeta in GradInputMeta() to obtain forward data "
              "types."
              "However, no TensorMeta is detected in bwd_out_meta_."));

      auto fwd_data_type = paddle::framework::TransToProtoVarType(
          slot_meta.GetTensorMeta().dtype);
      const paddle::Tensor& grad = slot_out_grads[rank_id];

      if (paddle::framework::IsComplexType(fwd_data_type)) continue;
      if (!grad.impl()) continue;

      // Only Handle Complex To Real for DenseTensor for now
      if (phi::DenseTensor::classof(grad.impl().get())) {
        phi::DenseTensor* grad_dense_tensor =
            static_cast<phi::DenseTensor*>(grad.impl().get());

        auto curr_data_type =
            paddle::framework::TransToProtoVarType(grad_dense_tensor->type());
        if (!paddle::framework::IsComplexType(curr_data_type)) continue;

        // Convert Complex GradOut to Real
        auto out = std::make_shared<phi::DenseTensor>();
        paddle::framework::TransComplexToReal(
            fwd_data_type, curr_data_type, *grad_dense_tensor, out.get());

        (*out_grads)[slot_id][rank_id].set_impl(out);
      } else if (phi::distributed::DistTensor::classof(grad.impl().get())) {
        auto grad_dense_tensor =
            static_cast<phi::distributed::DistTensor*>(grad.impl().get())
                ->value();

        auto curr_data_type =
            paddle::framework::TransToProtoVarType(grad_dense_tensor.type());
        if (!paddle::framework::IsComplexType(curr_data_type)) continue;
        if (grad_dense_tensor.dims().size() == -1) continue;

        // Convert Complex GradOut to Real
        auto out = std::make_shared<phi::DenseTensor>();
        paddle::framework::TransComplexToReal(
            fwd_data_type, curr_data_type, grad_dense_tensor, out.get());

        *(static_cast<phi::distributed::DistTensor*>(
              (*out_grads)[slot_id][rank_id].impl().get())
              ->unsafe_mutable_value()) = *(out.get());
      }
    }
  }
}

std::vector<std::shared_ptr<GradNodeBase>> GradNodeBase::NextFunctions() {
  std::vector<std::shared_ptr<GradNodeBase>> next_nodes;
  const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
      metas = OutputMeta();

  for (const auto& meta_list : metas) {
    for (const GradSlotMeta& meta : meta_list) {
      const auto& edge = meta.GetEdge();
      std::shared_ptr<GradNodeBase> next_node = edge.GetMutableGradNode();
      next_nodes.push_back(next_node);
    }
  }

  return next_nodes;
}

uintptr_t GradNodeBase::GetPtr() const {
  return reinterpret_cast<uintptr_t>(this);
}

int64_t GradNodeBase::RegisterNodePostHook(
    std::shared_ptr<NodePostHookBase>&& hook) {
  post_hooks_.emplace(next_post_hook_id_, std::move(hook));
  return next_post_hook_id_++;
}

bool GradNodeBase::RemoveNodePostHook(int64_t hook_id) {
  auto remove_cnt = post_hooks_.erase(hook_id);
  if (remove_cnt == 0) {
    return false;
  }
  return true;
}

bool GradNodeBase::HasNodePostHook() { return !post_hooks_.empty(); }

paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
GradNodeBase::ApplyNodePostHooks(
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>& grad_outputs,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>& grad_inputs) {
  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize> outs =
      grad_outputs;
  for (auto& iter : post_hooks_) {
    auto hook = iter.second;
    outs = (*hook)(outs, grad_inputs);
  }

  for (size_t i = 0; i < outs.size(); i++) {
    if (outs[i].empty() && (!grad_outputs[i].empty())) {
      outs[i].resize(grad_outputs[i].size());
    }

    for (size_t j = 0; j < outs[i].size(); j++) {
      if (!outs[i][j].defined() || !outs[i][j].initialized()) {
        outs[i][j] = grad_outputs[i][j];
      } else {
        CheckTensor(grad_outputs[i][j], outs[i][j]);
      }
    }
  }

  return outs;
}
}  // namespace egr
