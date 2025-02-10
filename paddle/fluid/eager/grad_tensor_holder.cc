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

#include "paddle/fluid/eager/grad_tensor_holder.h"

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace egr {

void GradTensorHolder::SetBufferSlotRankZeros(size_t slot_id, size_t rank) {
  // Set not grad var to zero and set stop gradient as default value: true
  buffer_[slot_id][rank] =
      paddle::experimental::zeros_like(buffer_[slot_id][rank]);
}

void GradTensorHolder::CopyValueFromTensor(size_t slot_id,
                                           size_t rank,
                                           const paddle::Tensor& t,
                                           bool fill_one) {
  // TODO(jiabin): We need to deal with empty input_buffer with slot size not
  // empty;
  PADDLE_ENFORCE(
      slot_id < buffer_.size(),
      common::errors::Fatal("Invalid slot_id for GradTensorHolder::add() "
                            "which exceeds size of buffer"));
  VLOG(6) << "Add Tensor for buffer_ slot: " << slot_id
          << ", size: " << buffer_[slot_id].size();
  if (buffer_[slot_id].empty()) {
    VLOG(6) << "Pass add Tensor for buffer_ slot: " << slot_id
            << " since its buffer_ is empty ";
    return;
  }
  PADDLE_ENFORCE(
      rank < buffer_[slot_id].size(),
      common::errors::Fatal(
          "Invalid rank for GradTensorHolder::add() which exceeds size "
          "of buffer slot %d, got slot size is: %d rank is: %d",
          slot_id,
          buffer_[slot_id].size(),
          rank));
  if (!fill_one) {
    paddle::Tensor& buffer_tensor = buffer_[slot_id][rank];
    if ((!buffer_tensor.defined() || !buffer_tensor.has_allocation())) {
      // Perform deep copy here
      buffer_tensor.copy_(t, t.place(), false);
      auto* meta = egr::EagerUtils::autograd_meta(&buffer_tensor);
      auto* origin_meta = egr::EagerUtils::nullable_autograd_meta(t);
      if (origin_meta) {
        auto grad_node = origin_meta->GetMutableGradNode();
        if (grad_node && grad_node.get()) {
          meta->SetGradNode(origin_meta->GetMutableGradNode());
        }
        meta->WeakGrad() = origin_meta->WeakGrad();
      }
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "Cannot copy grad_tensors' value to grad tensor holders,"
          "input buffer has already been initialized."));
    }
  } else {
    // Create new tensor->impl and fill it with 1.0
    if (t.defined()) {
      // Fill 1.0, use full to support complex, one_like don't support it.
      if (t.is_dense_tensor()) {  // NOLINT
        buffer_[slot_id][rank] =
            paddle::experimental::full(t.shape(), 1, t.dtype(), t.place());
      } else if (t.is_sparse_csr_tensor() || t.is_sparse_coo_tensor()) {
        buffer_[slot_id][rank] =
            paddle::experimental::sparse::full_like(t, 1, t.dtype());
      } else if (t.is_dist_tensor()) {
        auto init_grad =
            paddle::experimental::full(t.shape(), 1, t.dtype(), t.place());
        auto global_dense_t =
            std::static_pointer_cast<phi::DenseTensor>(init_grad.impl());
        auto dist_t =
            static_cast<phi::distributed::DistTensor*>(t.impl().get());
        auto dist_attr = dist_t->dist_attr();
        dist_attr.clean_partial_status();
        init_grad.set_impl(std::make_shared<phi::distributed::DistTensor>(
            global_dense_t, dist_attr));
        buffer_[slot_id][rank] = init_grad;
      } else {
        PADDLE_THROW(common::errors::Fatal(
            "Only Support DENSE_TENSOR, SPARSE_COO_TENSOR, SPARSE_CSR_TENSOR "
            "now."));
      }
    }
  }
  egr::EagerUtils::autograd_meta(&(buffer_[slot_id][rank]))
      ->SetStopGradient(false);
}

void GradTensorHolder::add(size_t slot_id,
                           size_t rank,
                           const paddle::Tensor& t,
                           bool create_graph) {
  if (!t.has_allocation()) {
    if (t.defined() && t.is_dist_tensor() &&
        phi::distributed::NeedComputationClipForPP(t.impl())) {
      // Pipeline parallel still needs to construct GradNode graph
      // to make DistTensor's global shape and DistAttr information flow.
      // Skip grad accumulation will cause GradTensor disconnect to next
      // GradNode.
      VLOG(3) << "Do accumulate for uninitialized Tensor " << t.name()
              << " as it's DistTensor and it needs computation clip for "
                 "pipeline parallel.";
    } else {
      VLOG(3) << "No need to do accumulate for uninitialized t.";
      return;
    }
  }  // TODO(jiabin): Remove this when we fix all kernel.

  PADDLE_ENFORCE(
      slot_id < buffer_.size(),
      common::errors::Fatal("Invalid slot_id for GradTensorHolder::add() "
                            "which exceeds size of buffer"));
  if (buffer_[slot_id].empty()) {
    VLOG(6) << "Pass add Tensor for buffer_ slot: " << slot_id
            << " since its buffer_ is empty ";
    return;
  }
  PADDLE_ENFORCE(
      rank < buffer_[slot_id].size(),
      common::errors::Fatal(
          "Invalid rank for GradTensorHolder::add() which exceeds size "
          "of buffer slot %d, got slot size is: %d rank is: %d",
          slot_id,
          buffer_[slot_id].size(),
          rank));

  paddle::Tensor& buffer_tensor = buffer_[slot_id][rank];
  // TODO(jiabin): Code below is ugly to divide which inner var we used,
  // remove framework::Variable
  // related code later.
  // This if statement is trying to test neither phi::Tensor nor
  // framework::Variable is initialized.
  if ((!buffer_tensor.defined() || !buffer_tensor.has_allocation())) {
    // Simply copy tensor->impl
    VLOG(6) << "Move Tensor for buffer_ slot: " << slot_id
            << ", size: " << buffer_[slot_id].size();
    buffer_tensor = t;
  } else {
    VLOG(6) << "Add Tensor for buffer_ slot: " << slot_id
            << ", size: " << buffer_[slot_id].size();
    // Accumulation
    PADDLE_ENFORCE_EQ(
        t.has_allocation(),
        true,
        common::errors::Fatal(
            "We can only accumulate tensor having allocation, but we "
            "got tensor: %s without allocation, please check you network "
            "and make sure it creates grads.",
            t.name()));

    if (t.is_dense_tensor()) {
      if (buffer_tensor.is_dense_tensor()) {
        if (create_graph || t.is_custom_device()) {
          buffer_tensor = add_ad_func(t, buffer_tensor);
        } else {
          paddle::imperative::TensorAdd<paddle::Tensor>(t, &buffer_tensor);
        }
      } else {
        // TODO(jiabin): Support Other TensorBase later
        // TODO(zhanlve): Replace SelectedRowsAddTensor with
        // add_dygraph_function once it's supported
        paddle::Tensor new_buffer(std::make_shared<phi::DenseTensor>(),
                                  "tmp_accumulator");
        paddle::imperative::SelectedRowsAddTensor(
            buffer_tensor, t, &new_buffer);
        buffer_tensor.set_impl(new_buffer.impl());
      }
    } else if (t.is_sparse_coo_tensor()) {
      auto t_sparse = std::dynamic_pointer_cast<phi::SparseCooTensor>(t.impl());
      paddle::Tensor t_values(
          std::make_shared<phi::DenseTensor>(t_sparse->non_zero_elements()));
      // In fact, the gradient of SparseTensor is still a SparseTensor
      if (buffer_tensor.is_sparse_coo_tensor()) {
        auto buffer_sparse = std::dynamic_pointer_cast<phi::SparseCooTensor>(
            buffer_tensor.impl());
        paddle::Tensor buffer_values(std::make_shared<phi::DenseTensor>(
            buffer_sparse->non_zero_elements()));
        if (create_graph || t.is_custom_device()) {
          buffer_values = add_ad_func(t_values, buffer_values);
        } else {
          paddle::imperative::TensorAdd<paddle::Tensor>(t_values,
                                                        &buffer_values);
        }
      }
    } else if (t.is_dist_tensor()) {
      buffer_tensor = add_ad_func(t, buffer_tensor);
    } else {
      // TODO(jiabin): Support Other TensorBase later
      // TODO(zhanlve): Replace SelectedRowsAddTensor with add_dygraph_function
      // once it's supported
      if (buffer_tensor.is_dense_tensor()) {
        paddle::imperative::SelectedRowsAddToTensor(t, &buffer_tensor);
      } else {
        buffer_tensor =
            std::move(*paddle::imperative::SelectedRowsMerge<paddle::Tensor>(
                t, buffer_tensor));
      }
    }
  }
}

}  // namespace egr
