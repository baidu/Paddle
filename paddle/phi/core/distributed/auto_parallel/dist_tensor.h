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

#pragma once

#include <memory>

#include "paddle/phi/common/reduce_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/placement_types.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {
class ReshardFunction;
class Shard;
class Partial;
class Replicate;

TensorDistAttr ToTensorDistAttr(const ProcessMesh& process_mesh,
                                const Placements& placements,
                                const DDim& dims);

Placements ToPlacements(const TensorDistAttr& dist_attr);

class DistTensor final
    : public phi::TensorBase,
      public phi::TypeInfoTraits<phi::TensorBase, DistTensor> {
 public:
  /// \brief Careful to create dist tensor using default constructor.
  /// this should only used in reshard for now, and the dist properties
  /// will be set by reshard later.
  DistTensor();

  /// \brief Construct a dist tensor based dtype.
  /// \param dtype The dtype of the current tensor.
  explicit DistTensor(phi::DataType dtype);

  /// \brief Construct a dist tensor based dense tensor.
  /// \param global_value The global dense tensor of the current tensor.
  /// \param dist_attr The distributed attributes of the current tensor.
  DistTensor(const std::shared_ptr<phi::DenseTensor>& global_value,
             const TensorDistAttr& dist_attr);

  /// \brief Construct a dist tensor based dense tensor.
  /// \param process_mesh The process mesh of the current tensor.
  /// \param placements The distributed placements of the current tensor.
  DistTensor(const std::shared_ptr<phi::DenseTensor>& global_value,
             const ProcessMesh& process_mesh,
             const Placements& placements);

  /// \brief Construct a dist tensor based local dense tensor.
  /// \param global_dims The global dim of the dist tensor.
  /// \param dist_attr The distributed attributes of the current tensor.
  DistTensor(const std::shared_ptr<phi::DenseTensor>& local_value,
             const DDim& global_dims,
             const TensorDistAttr& dist_attr);

  /// \brief Construct a dist tensor based local dense tensor.
  /// \param global_dims The global dim of the dist tensor.
  /// \param process_mesh The process mesh of the current tensor.
  /// \param placements The distributed placements of the current tensor.
  DistTensor(const std::shared_ptr<phi::DenseTensor>& local_value,
             const DDim& global_dims,
             const ProcessMesh& process_mesh,
             Placements  placements);

  /// \brief Construct a empty dist tensor (for infer spmd)
  /// \param dims The global dimension of the current Tensor.
  /// \param dist_attr The distributed attributes of the current tensor.
  DistTensor(const DDim& dims, const TensorDistAttr& dist_attr);

  /// \brief Destroy the tensor object and release exclusive resources.
  virtual ~DistTensor() = default;

  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "DistTensor"; }

  /// \brief Returns the global dims of the dist tensor.
  /// \return The global dims of the dist tensor.
  const DDim& dims() const override { return global_dims_; }

  /// \brief Set the global dims of the dist tensor.
  /// \return void
  void unsafe_set_dims(const DDim& dims);

  /// \brief Returns the dist attr of current dist tensor.
  /// \return The TensorDistAttr's const reference
  const TensorDistAttr& dist_attr() const { return dist_attr_; }

  /// \brief Returns the process_mesh of current dist tensor.
  /// \return The ProcessMesh's const reference
  const ProcessMesh& process_mesh() const { return process_mesh_; }

  /// \brief Returns the placements of current dist tensor.
  /// \return The Placements's const reference
  const Placements& placements() const { return placements_; }

  /// \brief Returns the num_shard of current dist tensor.
  /// \return int64_t
  int64_t num_shard() const {
    int64_t num_shard = 1;
    const auto& mesh_shape = process_mesh_.shape();
    for (size_t i = 0; i < placements_.size(); i++) {
      if (placements_[i]->is_shard()) {
        num_shard *= mesh_shape[i];
      }
    }
    return num_shard;
  }
  /// \brief Set the dist attr of current dist tensor.
  /// \return void
  void unsafe_set_dist_attr(const TensorDistAttr& dist_attr);

  /// \brief Returns the dense tensor value's const reference in dist tensor.
  /// \return The DenseTensor value's const reference
  const DenseTensor& value() const { return *value_; }

  /// \brief Returns the shared_ptr of dense tensor value's in dist tensor.
  /// \return The shared_ptr of dense tensor value
  std::shared_ptr<DenseTensor> shared_value() { return value_; }

  /// \brief Returns the mutable dense tensor value in dist tensor.
  /// \note If DenseTensor value is modified externally, the corresponding
  /// relationship between it and the current tensor's global dims and
  /// dist attr may be destroyed, which may introduce some subtle bugs,
  /// so you need to make sure to consider it thoroughly when using
  /// this method.
  /// \return The mutable pointer of DenseTensor value
  DenseTensor* unsafe_mutable_value() { return value_.get(); }

  /// \brief Returns the global dims of the dist tensor.
  /// \return The global dims of the dist tensor.
  const DDim& local_dims() const;

  /// \brief Test whether the holder is created.
  /// \return Whether the holder is created.
  bool has_allocation() const override;

  /// \brief Returns the global number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const override;

  /// \brief Test whether the dense tensor value's storage is allocated.
  /// \return Whether the dense tensor value's storage is allocated.
  bool initialized() const override;

  /// \brief Test whether the dense tensor value is defined.
  /// \return Whether the dense tensor value is defined.
  bool defined() const;

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const override;

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const override;

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const override;

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override;

  /// \brief Allocate memory with requested size from allocator.
  /// \return The mutable data pointer value of type T.
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;

  /// \brief Set the flag indicating whether to skip checking the process mesh.
  /// \note Currently only used for the MoE apis,
  /// it receives the inputs with different process meshes and outputs the dist
  /// tensor with global process mesh.
  /// \return void
  void unsafe_set_skip_check_mesh(bool skip);

  bool skip_check_mesh() const { return dist_attr_.skip_check_mesh(); }

  void clear();

 private:
  friend class ReshardFunction;

  // The global dimensions(shape), will move to DistTensorMeta
  DDim global_dims_;
  // The distributed attributes, will remove in the future
  TensorDistAttr dist_attr_;
  // The local DenseTensor value
  std::shared_ptr<DenseTensor> value_;

  ProcessMesh process_mesh_;

  Placements placements_;
};

}  // namespace distributed
}  // namespace phi
