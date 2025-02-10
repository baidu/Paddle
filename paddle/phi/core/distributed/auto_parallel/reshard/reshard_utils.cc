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

#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"

#include "glog/logging.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/core/enforce.h"

namespace phi::distributed {

namespace {
std::string GenUniqueCommKey(const std::vector<int64_t>& process_ids) {
  std::string unique_comm_key = "ReshardGroup";
  for (const auto& id : process_ids) {
    unique_comm_key += "/" + std::to_string(id);
  }
  return unique_comm_key;
}
}  // namespace

std::vector<int64_t> GetUnionProcessIds(std::vector<int64_t> in_process_ids,
                                        std::vector<int64_t> out_process_ids) {
  std::vector<int64_t> result;
  std::sort(in_process_ids.begin(), in_process_ids.end());
  std::sort(out_process_ids.begin(), out_process_ids.end());
  std::set_union(in_process_ids.begin(),
                 in_process_ids.end(),
                 out_process_ids.begin(),
                 out_process_ids.end(),
                 std::back_inserter(result));
  return result;
}

int64_t GetLocalRankInParticipate(const std::vector<int64_t>& process_ids,
                                  int64_t global_rank) {
  if (global_rank == -1) {
    global_rank = GetCurGlobalRank();
  }
  auto iter = std::find(process_ids.begin(), process_ids.end(), global_rank);
  PADDLE_ENFORCE_NE(
      iter,
      process_ids.end(),
      common::errors::NotFound(
          "Global rank %lld cannot be found in process_mesh", global_rank));
  return iter - process_ids.begin();
}

std::vector<int64_t> GetCurRankCoordInMesh(const ProcessMesh& process_mesh) {
  const auto& process_shape = process_mesh.shape();
  const auto& process_ids = process_mesh.process_ids();
  int64_t ndims_mesh = static_cast<int64_t>(process_shape.size());
  int64_t cur_global_rank = GetCurGlobalRank();

  VLOG(3) << "Searching current global rank " << cur_global_rank
          << " in process_mesh " << process_mesh;

  auto iter =
      std::find(process_ids.begin(), process_ids.end(), cur_global_rank);
  PADDLE_ENFORCE_NE(
      iter,
      process_ids.end(),
      common::errors::NotFound("Rank %lld cannot be found in process_mesh",
                               cur_global_rank));

  int64_t flat_idx_in_mesh = iter - process_ids.begin();

  std::vector<int64_t> coord(ndims_mesh, -1);
  for (int64_t i = ndims_mesh - 1; i >= 0; --i) {
    coord[i] = flat_idx_in_mesh % process_shape[i];
    flat_idx_in_mesh /= process_shape[i];
  }
  return coord;
}

CommContext* CreateOrGetCommContext(const DeviceContext& dev_ctx,
                                    const std::vector<int64_t>& process_ids) {
  std::string unique_comm_key = GenUniqueCommKey(process_ids);

  if (!CommContextManager::GetInstance().Has(unique_comm_key)) {
    int64_t world_size = static_cast<int64_t>(process_ids.size());
    int64_t rank = GetLocalRankInParticipate(process_ids);
    VLOG(3) << "local world size: " << world_size << " local rank: " << rank;

    auto store = CreateOrGetGlobalTCPStore();
    if (phi::CPUContext::classof(&dev_ctx)) {
#if defined(PADDLE_WITH_GLOO)
      CommContextManager::CreateGlooCommContext(store,
                                                unique_comm_key,
                                                static_cast<int>(rank),
                                                static_cast<int>(world_size));
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "Cannot use gloo on CPU, please turn PADDLE_WITH_GLOO flag on."));
#endif
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_ROCM)
    else if (phi::GPUContext::classof(&dev_ctx)) {  // NOLINT
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      CommContextManager::CreateNCCLCommContext(store,
                                                unique_comm_key,
                                                static_cast<int>(rank),
                                                static_cast<int>(world_size));
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "Cannot use nccl on GPU, please turn WITH_NCCL flag on."));
#endif
    }
#elif defined(PADDLE_WITH_XPU)
    else if (phi::XPUContext::classof(&dev_ctx)) {  // NOLINT
#if defined(PADDLE_WITH_XPU_BKCL)
      CommContextManager::CreateBKCLCommContext(store,
                                                unique_comm_key,
                                                static_cast<int>(rank),
                                                static_cast<int>(world_size));
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "Cannot use xpu on GPU, please turn WITH_XPU_BKCL flag on."));
#endif
    }
#elif defined(PADDLE_WITH_CUSTOM_DEVICE)
    else if (phi::CustomContext::classof(&dev_ctx)) {  // NOLINT
      CommContextManager::CreateXCCLCommContext(
          store, unique_comm_key, dev_ctx.GetPlace(), rank, world_size);
    }
#endif
    else {  // NOLINT
      PADDLE_THROW(common::errors::Unimplemented(
          "CommContext is only supported CPU, GPU, XPU, and CustomDevice."));
    }
  }

  auto* comm_context = CommContextManager::GetInstance().Get(unique_comm_key);
  return comm_context;
}

std::map<int, int64_t> GetSplitAxisWithDimsMapping(
    const std::vector<int64_t>& dims_mapping) {
  std::map<int, int64_t> split_axis_to_mesh_axis;
  for (size_t i = 0; i < dims_mapping.size(); ++i) {
    if (dims_mapping[i] != -1) {
      split_axis_to_mesh_axis.emplace(i, dims_mapping[i]);
    }
  }
  return split_axis_to_mesh_axis;
}

std::vector<int64_t> BalancedSplit(int64_t total_nums, int64_t num_of_pieces) {
  bool has_remainder = (total_nums % num_of_pieces != 0);
  std::vector<int64_t> result(num_of_pieces,
                              (total_nums + num_of_pieces - 1) / num_of_pieces);
  if (has_remainder) {
    int64_t& last_value = result.back();
    last_value = last_value - (last_value * num_of_pieces - total_nums);
  }
  return result;
}

bool IsCurRankInMesh(const ProcessMesh& process_mesh) {
  int64_t cur_global_rank = GetCurGlobalRank();
  const auto& process_ids = process_mesh.process_ids();
  return (std::find(process_ids.begin(), process_ids.end(), cur_global_rank) !=
          process_ids.end());
}

// Only Input is DistTensor and current device id isn't in DistTensor's mesh
// will return true.
bool NeedComputationClipForPP(
    const std::shared_ptr<phi::TensorBase>& tensor_impl) {
  PADDLE_ENFORCE_EQ(
      phi::distributed::DistTensor::classof(tensor_impl.get()),
      true,
      common::errors::InvalidArgument(
          "The input tensor of NeedComputationClipForPP should be "
          "``phi::distributed::DistTensor``. "
          "However it's %s",
          typeid(tensor_impl.get()).name()));
  return !IsCurRankInMesh(
      std::static_pointer_cast<phi::distributed::DistTensor>(tensor_impl)
          ->dist_attr()
          .process_mesh());
}

Place GetDefaultPlace() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::backends::gpu::GetGPUDeviceCount() >= 0) {
    return paddle::DefaultGPUPlace();
  }
#endif
  return paddle::CPUPlace();
}

phi::DeviceContext* GetDistTensorDeviceContext(
    phi::distributed::DistTensor* input) {
  // TODO(GhostScreaming): pipeline parallel may create an undefined middle grad
  // tensor. In such case, we need to get default place.
  auto place =
      input && input->initialized() ? input->place() : GetDefaultPlace();
  return phi::DeviceContextPool::Instance().Get(place);
}

phi::DDim InferShapeForReshardFromReplicate(
    const std::shared_ptr<phi::DenseTensor>& global_value,
    const TensorDistAttr& dist_attr) {
  phi::DDim out_dim = global_value->dims();
  auto coord_id = GetCurRankCoordInMesh(dist_attr.process_mesh());
  for (int tensor_axis = 0; tensor_axis < global_value->dims().size();
       ++tensor_axis) {
    if (dist_attr.is_shard(-1, tensor_axis)) {
      for (int mesh_axis = 0; mesh_axis < dist_attr.process_mesh().ndim();
           ++mesh_axis) {
        if (dist_attr.is_shard(mesh_axis, tensor_axis)) {
          // handle the shard axis
          int64_t global_shape = out_dim[tensor_axis];
          int64_t mesh_size = dist_attr.process_mesh().dim_size(mesh_axis);
          auto balance_shard = BalancedSplit(global_shape, mesh_size);
          out_dim[tensor_axis] = balance_shard[coord_id[mesh_axis]];
        }
      }
    }
  }
  return out_dim;
}

// 1. Get all the sub meshes of global_mesh
// e.g. global_mesh = [[1, 2], [3, 4]], out_mesh = [1, 2] and [3, 4]
//      global_mesh = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
//      out_mesh = [[1, 2], [3, 4]] and [[5, 6], [7, 8]]
std::vector<ProcessMesh> GetSubMeshes(const ProcessMesh& process_mesh) {
  const std::vector<int64_t>& shape = process_mesh.shape();
  const std::vector<int64_t>& process_ids = process_mesh.process_ids();
  const std::vector<std::string>& dim_names = process_mesh.dim_names();
  int64_t total_process_num = process_ids.size();
  int64_t sub_process_num = total_process_num / shape[0];
  std::vector<int64_t> sub_process_mesh_shape(shape.begin() + 1, shape.end());
  std::vector<std::string> sub_process_mesh_dim_names(dim_names.begin() + 1,
                                                      dim_names.end());

  std::vector<ProcessMesh> sub_process_meshes;
  for (int i = 0; i < shape[0]; ++i) {
    int64_t start_position = i * sub_process_num;
    int64_t end_position = start_position + sub_process_num;
    std::vector<int64_t> sub_process_ids(process_ids.begin() + start_position,
                                         process_ids.begin() + end_position);

    sub_process_meshes.emplace_back(
        sub_process_mesh_shape, sub_process_ids, sub_process_mesh_dim_names);
  }
  return sub_process_meshes;
}

bool IsSubMesh(const ProcessMesh& global_mesh, const ProcessMesh& sub_mesh) {
  std::vector<ProcessMesh> sub_process_meshes = GetSubMeshes(global_mesh);
  for (const ProcessMesh& mesh : sub_process_meshes) {
    if (mesh == sub_mesh) {
      return true;
    }
  }
  return false;
}

}  // namespace phi::distributed
