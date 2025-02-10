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

#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_x_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/add_n_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/p_recv_kernel.h"
#include "paddle/phi/kernels/p_send_kernel.h"
#include "paddle/phi/kernels/split_kernel.h"

namespace phi::distributed {

bool RToXExpandReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_replicated());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.process_ids().size() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.process_ids().size() != 1);

  return true;
}

void RToXExpandReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                                     const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr,
                                     DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& in_dist_attr = in.dist_attr();
  const auto& out_dims_mapping = out_dist_attr.dims_mapping();
  const auto& in_mesh = in_dist_attr.process_mesh();
  const auto& out_mesh = out_dist_attr.process_mesh();
  const auto& in_process_ids = in_mesh.process_ids();
  const auto& out_process_ids = out_mesh.process_ids();
  int64_t cur_global_rank = GetCurGlobalRank();
  int64_t root_rank = in_process_ids[0];
  auto all_process_ids = GetUnionProcessIds(in_process_ids, out_process_ids);
  auto dtype = in.dtype();
  const auto& out_partial_status = out_dist_attr.partial_status();
  bool cur_rank_in_out_mesh =
      (std::find(out_process_ids.begin(),
                 out_process_ids.end(),
                 cur_global_rank) != out_process_ids.end());
  DenseTensor result_value;

  if (root_rank == cur_global_rank) {
    for (const auto& out_process_id : out_process_ids) {
      if (out_process_id != root_rank) {
#if defined(PADDLE_WITH_XPU)
        PADDLE_THROW(::common::errors::Unimplemented(
            "Not supported PSendKernel on xpu yet."));
#else
        RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                                  PSendKernel,
                                  dtype,
                                  all_process_ids,
                                  in.value(),
                                  out_process_id,
                                  /*dynamic_shape=*/true);
#endif
      }
    }
    if (cur_rank_in_out_mesh) {
      result_value = in.value();
    }
  } else {
#if defined(PADDLE_WITH_XPU)
    PADDLE_THROW(
        ::common::errors::Unimplemented("Not supported PRecv on xpu yet."));
#else
    RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                              PRecv,
                              dtype,
                              all_process_ids,
                              root_rank,
                              {} /*out_shape*/,
                              /*dynamic_shape=*/true,
                              &result_value);
#endif
  }

  if (cur_rank_in_out_mesh) {
    if (out_dist_attr.is_partial()) {
      auto out_reduce_type = out_partial_status.at(0);
      if (out_reduce_type == ReduceType::kRedSum &&
          cur_global_rank != out_process_ids[0]) {
        IntArray shape(result_value.dims().Get(), result_value.dims().size());
        RESHARD_FUNCTOR(dev_ctx, Full, dtype, shape, 0, &result_value);
      }
      SetValue(out, result_value);
    } else if (out_dist_attr.is_shard()) {
      std::map<int, int64_t> split_axis_to_mesh_axis =
          GetSplitAxisWithDimsMapping(out_dims_mapping);
      std::vector<int64_t> coord_in_mesh = GetCurRankCoordInMesh(out_mesh);

      int split_axis = split_axis_to_mesh_axis.begin()->first;
      int64_t mesh_axis = split_axis_to_mesh_axis.begin()->second;
      int64_t num_of_process = out_mesh.shape()[mesh_axis];

      std::vector<int64_t> split_num_vec =
          BalancedSplit(in.dims()[split_axis], num_of_process);
      IntArray sections(split_num_vec);

      std::vector<DenseTensor> split_out_vec;
      RESHARD_FUNCTOR(dev_ctx,
                      Split,
                      dtype,
                      result_value,
                      sections,
                      split_axis,
                      &split_out_vec);

      SetValue(out, split_out_vec[coord_in_mesh[mesh_axis]]);
    } else {
      SetValue(out, result_value);
    }
    SetDistProps(out, in.dims(), out_dist_attr);
  }
}

}  // namespace phi::distributed
