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

#include "paddle/phi/core/distributed/auto_parallel/reshard/x_to_r_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/add_n_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/p_recv_kernel.h"
#include "paddle/phi/kernels/p_send_kernel.h"

namespace phi::distributed {

bool XToRShrinkReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_replicated());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.process_ids().size() != 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.process_ids().size() == 1);

  return true;
}

void XToRShrinkReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                                     const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr,
                                     DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_dims_mapping = in_dist_attr.dims_mapping();
  const auto& in_mesh = in_dist_attr.process_mesh();
  const auto& out_mesh = out_dist_attr.process_mesh();
  const auto& in_process_ids = in_mesh.process_ids();
  const auto& out_process_ids = out_mesh.process_ids();
  int64_t cur_global_rank = GetCurGlobalRank();
  int64_t root_rank = out_process_ids[0];
  auto dtype = in.dtype();
  const auto& in_partial_status = in_dist_attr.partial_status();
  auto all_process_ids = GetUnionProcessIds(in_process_ids, out_process_ids);
  std::unordered_map<int64_t, DenseTensor> rank_to_result;

  // Step 1: other ranks need to send value to the root
  if (!in_dist_attr.is_replicated()) {
    if (cur_global_rank != root_rank) {
      // send dense tensor to root
#if defined(PADDLE_WITH_XPU)
      PADDLE_THROW(::common::errors::Unimplemented(
          "Not supported PSendKernel  on xpu yet."));
#else
      RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                                PSendKernel,
                                dtype,
                                all_process_ids,
                                in.value(),
                                root_rank,
                                /*dynamic_shape=*/true);
#endif
    } else {
      for (size_t i = 0; i < all_process_ids.size(); ++i) {
        if (all_process_ids[i] != root_rank) {
          rank_to_result.emplace(all_process_ids[i], DenseTensor());
#if defined(PADDLE_WITH_XPU)
          PADDLE_THROW(::common::errors::Unimplemented(
              "Not supported PRecv on xpu yet."));
#else
          RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                                    PRecv,
                                    dtype,
                                    all_process_ids,
                                    all_process_ids[i],
                                    {} /*out_shape*/,
                                    /*dynamic_shape=*/true,
                                    &rank_to_result[all_process_ids[i]]);
#endif
        }
      }
    }
  }

  // Step 2: concat or reduce based on dist attr
  if (cur_global_rank == root_rank) {
    std::vector<const DenseTensor*> input_vec;
    for (const auto& in_process_id : in_process_ids) {
      if (in_process_id == cur_global_rank) {
        input_vec.emplace_back(&(in.value()));
      } else {
        input_vec.emplace_back(&(rank_to_result[in_process_id]));
      }
    }
    if (in_dist_attr.is_shard()) {
      int split_axis =
          GetSplitAxisWithDimsMapping(in_dims_mapping).begin()->first;
      RESHARD_FUNCTOR(
          dev_ctx, Concat, dtype, input_vec, split_axis, GetMutableTensor(out));
    } else if (in_dist_attr.is_partial()) {
      auto in_reduce_type = in_partial_status.at(0);
      if (in_reduce_type == ReduceType::kRedSum) {
        DenseTensor result_add_out = *input_vec[0];
        for (size_t i = 1; i < input_vec.size(); ++i) {
          RESHARD_FUNCTOR(dev_ctx,
                          Add,
                          dtype,
                          *input_vec[i],
                          result_add_out,
                          &result_add_out);
        }
        SetValue(out, result_add_out);
      } else {
        PADDLE_THROW(common::errors::Unavailable(
            "The reduce type is not supported, will be supported soon."));
      }
    } else {
      SetValue(out, in.value());
    }
    SetDistProps(out, in.dims(), out_dist_attr);
  }
}

}  // namespace phi::distributed
