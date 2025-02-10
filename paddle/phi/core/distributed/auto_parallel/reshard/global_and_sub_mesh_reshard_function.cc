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

#include "paddle/phi/core/distributed/auto_parallel/reshard/global_and_sub_mesh_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/p_recv_kernel.h"
#include "paddle/phi/kernels/p_send_kernel.h"

namespace phi::distributed {

bool GlobalToSubMeshReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const TensorDistAttr& in_dist_attr = in.dist_attr();

  const ProcessMesh& in_process_mesh = in_dist_attr.process_mesh();
  const ProcessMesh& out_process_mesh = out_dist_attr.process_mesh();

  int sub_mesh_dim = SubMeshDim(in_process_mesh, out_process_mesh);
  RESHARD_SHORTCUT_IF_FALSE(sub_mesh_dim != -1);
  // 1. the splitted dimension must be replicated
  // 2. out mesh is the value of a certain dimension of global mesh
  // e.g. global_mesh = [[1, 2], [3, 4]], out_mesh = [1, 2] or [3, 4]
  //      global_mesh = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
  //      out_mesh = [[1, 2], [3, 4]] or [[5, 6], [7, 8]]
  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_replicated(sub_mesh_dim));

  return true;
}

void GlobalToSubMeshReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                                          const DistTensor& in,
                                          const TensorDistAttr& out_dist_attr,
                                          DistTensor* out) {
  VLOG(3) << "Call GlobalToSubMeshReshardFunction Eval";
  const DenseTensor& in_dense_value = in.value();
  const ProcessMesh& out_process_mesh = out_dist_attr.process_mesh();
  if (IsCurRankInMesh(out_process_mesh)) {
    SetValue(out, in_dense_value);
  } else {
    *(out->unsafe_mutable_value()) =
        phi::DenseTensor(std::make_shared<phi::Allocation>(
                             nullptr, 0, phi::distributed::GetDefaultPlace()),
                         in.value().meta());
  }
  SetDistProps(out, in.dims(), out_dist_attr);
}

bool SubMeshToGlobalReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const TensorDistAttr& in_dist_attr = in.dist_attr();
  const ProcessMesh& in_process_mesh = in_dist_attr.process_mesh();
  const ProcessMesh& out_process_mesh = out_dist_attr.process_mesh();

  int sub_mesh_dim = SubMeshDim(out_process_mesh, in_process_mesh);
  RESHARD_SHORTCUT_IF_FALSE(sub_mesh_dim != -1);
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_replicated(sub_mesh_dim));

  return true;
}

void SubMeshToGlobalReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                                          const DistTensor& in,
                                          const TensorDistAttr& out_dist_attr,
                                          DistTensor* out) {
  VLOG(3) << "Call SubMeshToGlobalReshardFunction Eval";
#if defined(PADDLE_WITH_XPU)
  PADDLE_THROW(::common::errors::Unimplemented(
      "Not supported PSendKernel/PRecv on xpu yet."));
#else
  const TensorDistAttr& in_dist_attr = in.dist_attr();
  const ProcessMesh& in_process_mesh = in_dist_attr.process_mesh();
  const ProcessMesh& out_process_mesh = out_dist_attr.process_mesh();

  int sub_mesh_dim = SubMeshDim(out_process_mesh, in_process_mesh);
  std::vector<ProcessMesh> sub_process_meshes =
      SplitMesh(out_process_mesh, sub_mesh_dim);
  const std::vector<int64_t>& in_process_ids = in_process_mesh.process_ids();
  const std::vector<int64_t>& out_process_ids = out_process_mesh.process_ids();
  std::unordered_map<int64_t, std::vector<int64_t>> send2recv_map;
  std::unordered_map<int64_t, int64_t> recv2send_map;

  for (const ProcessMesh& sub_mesh : sub_process_meshes) {
    if (mesh_equal_ignore_shape1(sub_mesh, in_process_mesh, sub_mesh_dim)) {
      continue;
    }
    const std::vector<int64_t>& sub_process_ids = sub_mesh.process_ids();
    for (size_t i = 0; i < sub_process_ids.size(); ++i) {
      int64_t send_id = in_process_ids[i];
      send2recv_map[send_id].push_back(sub_process_ids[i]);
      recv2send_map[sub_process_ids[i]] = send_id;
    }
  }

  std::vector<int64_t> all_process_ids =
      GetUnionProcessIds(in_process_ids, out_process_ids);
  int64_t cur_global_rank = GetCurGlobalRank();
  DataType dtype = in.dtype();
  if (IsCurRankInMesh(in_process_mesh)) {
    const DenseTensor& in_dense_value = in.value();
    std::vector<int64_t>& recv_vec = send2recv_map[cur_global_rank];
    for (int64_t recv_id : recv_vec) {
      RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                                PSendKernel,
                                dtype,
                                all_process_ids,
                                in_dense_value,
                                recv_id,
                                true /*dynamic_shape*/);
    }
    SetValue(out, in_dense_value);
  } else {
    int64_t send_id = recv2send_map[cur_global_rank];
    RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                              PRecv,
                              dtype,
                              all_process_ids,
                              send_id,
                              {} /*out_shape*/,
                              true /*dynamic_shape*/,
                              GetMutableTensor(out));
  }
  SetDistProps(out, in.dims(), out_dist_attr);
#endif
}

}  // namespace phi::distributed
