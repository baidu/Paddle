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

#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"

#include <utility>

#include "glog/logging.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function_registry.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/store/store_utils.h"

namespace phi::distributed {

inline void check_defined(const DistTensor& dist_tensor,
                          std::string method_hint) {
  PADDLE_ENFORCE_EQ(
      dist_tensor.defined(),
      true,
      common::errors::Unimplemented(
          "DistTensor is not defined yet when `%s` method is called.",
          method_hint));
}

TensorDistAttr ToTensorDistAttr(const ProcessMesh& process_mesh,
                                const Placements& placements,
                                const DDim& dims) {
  TensorDistAttr dist_attr(vectorize(dims));
  // Step1: set process_mesh
  dist_attr.set_process_mesh(process_mesh);

  // Step2: set dim_mapping
  int64_t ndim = dims.size();
  std::vector<int64_t> dim_map(ndim, -1);
  for (size_t i = 0; i < placements.size(); i++) {
    auto& placement = placements[i];
    if (placement->is_shard()) {
      auto shard_dim = dynamic_cast<const Shard&>(*placement).get_dim();
      if (dim_map[shard_dim] != -1) {
        LOG(WARNING) << "WARNING: Tensor dim " << shard_dim
                     << " is already sharded on "
                     << "mesh dim" << dim_map[shard_dim]
                     << ". Sharding a tensor dim with "
                     << "multiple mesh dim is not supported yet.";
      }
      // PADDLE_ENFORCE_EQ(
      //     dim_map[shard_dim],
      //     -1,
      //     common::errors::InvalidArgument(
      //         "Tensor dim %lld is already sharded on mesh dim %lld,"
      //         " DistTensor operator implementation does not support things "
      //         "like hybrid"
      //         " sharding strategies yet (i.e. [Shard(0), Shard(0)])",
      //         shard_dim,
      //         dim_map[shard_dim]));
      dim_map[shard_dim] = i;
    }
  }
  dist_attr.set_dims_mapping(dim_map);

  // Step3: set partial_status
  paddle::flat_hash_map<int64_t, ReduceType> partial_status;
  for (size_t i = 0; i < placements.size(); ++i) {
    auto& p = placements[i];
    if (p->is_partial()) {
      partial_status.insert({i, dynamic_cast<Partial&>(*p).get_reduce_type()});
    }
  }
  dist_attr.set_partial_status(partial_status);

  // Step4: mark annotated
  dist_attr.mark_annotated("process_mesh");
  dist_attr.mark_annotated("dims_mapping");
  return dist_attr;
}

Placements ToPlacements(const TensorDistAttr& dist_attr) {
  auto& process_mesh = dist_attr.process_mesh();
  Placements placements;
  placements.resize(process_mesh.ndim(), std::make_shared<Replicate>());

  auto& partial_status = dist_attr.partial_status();
  for (const auto& pair : partial_status) {
    placements[pair.first] = std::make_shared<Partial>(pair.second);
  }

  auto& dim_mapping = dist_attr.dims_mapping();
  for (size_t i = 0; i < dim_mapping.size(); ++i) {
    auto& mesh_id = dim_mapping[i];
    if (mesh_id >= 0) {
      auto& p = placements[mesh_id];

      if (p->is_shard()) {
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "ProcessMesh dimension can't be mapped to two  dimension of the "
            "same tensor: {%d} and {%d}",
            i,
            dynamic_cast<Shard&>(*p).get_dim()));
      } else if (p->is_partial()) {
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "ProcessMesh dimension {%d} cannot be both shard and partial!",
            mesh_id));
      }
      placements[mesh_id] = std::make_shared<Shard>(i);
    }
  }
  return placements;
}

DistTensor::DistTensor() : value_(std::make_shared<DenseTensor>()) {}

DistTensor::DistTensor(phi::DataType dtype)
    : value_(std::make_shared<DenseTensor>(dtype)) {}

DistTensor::DistTensor(const std::shared_ptr<phi::DenseTensor>& global_value,
                       const TensorDistAttr& dist_attr)
    : global_dims_(global_value->dims()), dist_attr_(dist_attr) {
  process_mesh_ = dist_attr_.process_mesh();
  placements_ = ToPlacements(dist_attr);

  // If the current rank doesn't in process_mesh, we should create an
  // uninitialized tensor only with tensor_meta.
  if (IsCurRankInMesh(dist_attr.process_mesh())) {
    if (!dist_attr.is_replicated()) {
      value_ = std::make_shared<DenseTensor>();
      // 1. create replicated global tensor
      TensorDistAttr replicated_dist_attr(
          common::vectorize(global_value->dims()));
      replicated_dist_attr.set_process_mesh(dist_attr.process_mesh());
      DistTensor replicated_tensor(global_value, replicated_dist_attr);

      // 2. reshard from replicated to other state
      VLOG(4) << "Reshard tensor: "
              << paddle::experimental::ReshardDebugInfo(replicated_tensor,
                                                        dist_attr);
      auto* func = ChooseProperReshardFunction(replicated_tensor, dist_attr);
      auto* dev_ctx = DeviceContextPool::Instance().Get(global_value->place());
      func->Eval(dev_ctx, replicated_tensor, dist_attr, this);
    } else {
      value_ = global_value;
    }
  } else {
    value_ = std::make_shared<DenseTensor>(
        std::make_shared<phi::Allocation>(nullptr, 0, global_value->place()),
        phi::DenseTensorMeta(global_value->meta()));
  }
}

DistTensor::DistTensor(const std::shared_ptr<phi::DenseTensor>& local_value,
                       const DDim& global_dims,
                       const ProcessMesh& process_mesh,
                       Placements  placements)
    : global_dims_(global_dims),
      process_mesh_(process_mesh),
      placements_(std::move(placements)) {
  dist_attr_ = ToTensorDistAttr(process_mesh_, placements_, global_dims_);
  if (IsCurRankInMesh(process_mesh)) {
    value_ = local_value;
  } else {
    value_ = std::make_shared<DenseTensor>(
        std::make_shared<phi::Allocation>(nullptr, 0, local_value->place()),
        phi::DenseTensorMeta(local_value->dtype(), phi::make_ddim({0})));
  }
}

DistTensor::DistTensor(const std::shared_ptr<phi::DenseTensor>& local_value,
                       const DDim& global_dims,
                       const TensorDistAttr& dist_attr)
    : global_dims_(global_dims), dist_attr_(dist_attr) {
  process_mesh_ = dist_attr_.process_mesh();
  placements_ = ToPlacements(dist_attr);
  if (IsCurRankInMesh(process_mesh_)) {
    value_ = local_value;
  } else {
    value_ = std::make_shared<DenseTensor>(
        std::make_shared<phi::Allocation>(nullptr, 0, local_value->place()),
        phi::DenseTensorMeta(local_value->dtype(), global_dims_));
  }
}

DistTensor::DistTensor(const std::shared_ptr<phi::DenseTensor>& global_value,
                       const ProcessMesh& process_mesh,
                       const Placements& placements)
    : global_dims_(global_value->dims()) {
  process_mesh_ = process_mesh;
  placements_ = placements;
  // If the dims.size() == -1, the dims=[0] by default, which is not consistent
  // and will cause ToTensorDistAttr‘s error.
  if (global_dims_ == DDim()) {
    global_dims_ = phi::make_ddim({});
  }
  dist_attr_ = ToTensorDistAttr(process_mesh_, placements_, global_dims_);

  // If the current rank doesn't in process_mesh, we should create an
  // uninitialized tensor only with dist_tensor_meta_.
  if (IsCurRankInMesh(process_mesh)) {
    if (!dist_attr_.is_replicated()) {
      if (global_value->initialized()) {
        value_ = std::make_shared<DenseTensor>();
        // 1. create replicated global tensor
        TensorDistAttr replicated_dist_attr(
            common::vectorize(global_value->dims()));
        replicated_dist_attr.set_process_mesh(process_mesh);
        DistTensor replicated_tensor(global_value, replicated_dist_attr);

        // 2. reshard from replicated to other state
        VLOG(4) << "Reshard tensor: "
                << paddle::experimental::ReshardDebugInfo(replicated_tensor,
                                                          dist_attr_);
        auto* func = ChooseProperReshardFunction(replicated_tensor, dist_attr_);
        auto* dev_ctx =
            DeviceContextPool::Instance().Get(global_value->place());
        func->Eval(dev_ctx, replicated_tensor, dist_attr_, this);
      } else {
        // For lazy init, the global value is an uninitialized tensor.
        // Just infer the local shape of the dist tensor.
        value_ = global_value;
        value_->Resize(
            InferShapeForReshardFromReplicate(global_value, dist_attr_));
      }
    } else {
      value_ = global_value;
    }
  } else {
    value_ = std::make_shared<DenseTensor>(
        std::make_shared<phi::Allocation>(nullptr, 0, global_value->place()),
        phi::DenseTensorMeta(global_value->meta()));
  }
}

DistTensor::DistTensor(const DDim& dims, const TensorDistAttr& dist_attr)
    : global_dims_(dims),
      dist_attr_(dist_attr),
      value_(std::make_shared<DenseTensor>()) {
  process_mesh_ = dist_attr.process_mesh();
  placements_ = ToPlacements(dist_attr);
}

void DistTensor::unsafe_set_dims(const DDim& dims) {
  if (this->initialized()) {
    VLOG(6) << "You try to set an initialized DistTensor's global dims. "
               "Make sure you are aware of where you change its dims.";
  }
  global_dims_ = dims;
}

void DistTensor::unsafe_set_dist_attr(const TensorDistAttr& dist_attr) {
  if (this->initialized()) {
    VLOG(6) << "You try to set an initialized DistTensor's dist attr. "
               "Make sure you are aware of where you change its dist attr.";
  }
  dist_attr_ = dist_attr;
  process_mesh_ = dist_attr.process_mesh();
  placements_ = ToPlacements(dist_attr);
}

int64_t DistTensor::numel() const {
  // DistTensor with uninitialized local tensor can
  // also have numel.
  return product(global_dims_);
}

const DDim& DistTensor::local_dims() const {
  check_defined(*this, "local_dims");
  return value_->dims();
}

bool DistTensor::valid() const {
  check_defined(*this, "valid");
  return value_->valid();
}

bool DistTensor::defined() const { return value_->holder_ != nullptr; }

bool DistTensor::has_allocation() const {
  return value_->holder_ != nullptr && (value_->holder_->ptr() || numel() != 0);
}

bool DistTensor::initialized() const {
  return value_->holder_ != nullptr && value_->holder_->ptr();
}

DataType DistTensor::dtype() const {
  // DistTensor with uninitialized local tensor can
  // also have dtype.
  return value_->dtype();
}

DataLayout DistTensor::layout() const {
  // DistTensor with uninitialized local tensor can
  // also have layout.
  return value_->layout();
}

const Place& DistTensor::place() const {
  check_defined(*this, "place");
  return value_->holder_->place();
}

void* DistTensor::AllocateFrom(Allocator* allocator,
                               DataType dtype,
                               size_t requested_size,
                               bool fake_alloc) {
  PADDLE_THROW(common::errors::Unavailable(
      "The DistTensor Cannot allocate memory directly and needs to perform "
      "memory operations through its DenseTensor value."));
  return nullptr;
}

void DistTensor::unsafe_set_skip_check_mesh(bool skip) {
  VLOG(6) << "You try to set an initialized DistTensor's dist attr. "
             "Make sure you are aware of where you change its dist attr.";
  dist_attr_.set_skip_check_mesh(skip);
}

void DistTensor::clear() {
  if (value_) {
    value_->clear();
  }
}

}  // namespace phi::distributed
