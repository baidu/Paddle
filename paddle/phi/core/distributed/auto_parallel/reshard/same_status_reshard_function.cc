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

#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"

#include <algorithm>

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/p_recv_kernel.h"
#include "paddle/phi/kernels/p_send_kernel.h"

namespace phi::distributed {

bool SameStatusReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.dims_mapping() ==
                            out_dist_attr.dims_mapping());
  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.partial_dims() ==
                            out_dist_attr.partial_dims());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh != out_process_mesh);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.shape() ==
                            out_process_mesh.shape());

  return true;
}

void SameStatusReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                                     const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr,
                                     DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& in_process_ids = in_process_mesh.process_ids();
  const auto& out_process_mesh = out_dist_attr.process_mesh();
  const auto& out_process_ids = out_process_mesh.process_ids();
  auto all_process_ids = GetUnionProcessIds(in_process_ids, out_process_ids);

  // TODO(GhostScreaming): After cross-mesh reshard, current device may
  // needs to execute next layer. When it construct next layer's backward
  // graph, out->place() will be called such as in SetGradOutMeta method. As
  // a result, out can't be undefined. Try to allocate a zero-memory value
  // for out. Following send/recv will cover this empty DenseTensor
  // construction.
  VLOG(3) << "Same_status_reshard_function create an empty DenseTensor for "
             "cross-mesh DistTensor.";
  *(out->unsafe_mutable_value()) =
      phi::DenseTensor(std::make_shared<phi::Allocation>(
                           nullptr, 0, phi::distributed::GetDefaultPlace()),
                       in.value().meta());

  std::vector<std::pair<int64_t, int64_t>> p2p_pair;
  for (size_t i = 0; i < out_process_ids.size(); ++i) {
    p2p_pair.emplace_back(in_process_ids[i], out_process_ids[i]);
  }
  int64_t cur_global_rank = GetCurGlobalRank();
  for (const auto& iter : p2p_pair) {
    int64_t src = iter.first;
    int64_t dst = iter.second;
    if (src == cur_global_rank) {
#if defined(PADDLE_WITH_XPU)
      PADDLE_THROW(::common::errors::Unimplemented(
          "Not supported PSendKernel on xpu yet."));
#else
      VLOG(3) << "Send from src " << src << " to dst " << dst;
      int64_t dst_local_rank = GetLocalRankInParticipate(all_process_ids, dst);
      // Since send kernel only has input, so we don't need to infermeta
      // actually. According to this reason, just use the kernel directly.
      RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                                PSendKernel,
                                in.dtype(),
                                all_process_ids,
                                in.value(),
                                dst_local_rank,
                                /*dynamic_shape=*/true);
      // TODO(liyurui): Use dynamic shape will lead to poor performance, but we
      // don't have any other good idea now. For the following reasons:
      // 1. We can not ensure the meta being right deduce by the infermeta.
      // 2. The meta of some kernels can't decide in compile time.
      // 3. DenseTensor with empty value only need infermeta and skip the real
      // kernel execution.
#endif
    } else if (dst == cur_global_rank) {
#if defined(PADDLE_WITH_XPU)
      PADDLE_THROW(::common::errors::Unimplemented(
          "Not supported PRecvKernel on xpu yet."));
#else
      VLOG(3) << "Recv from src " << src << " to dst " << dst;
      int64_t src_local_rank = GetLocalRankInParticipate(all_process_ids, src);
      RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                                PRecv,
                                in.dtype(),
                                all_process_ids,
                                src_local_rank,
                                {} /*out_shape*/,
                                /*dynamic_shape=*/true,
                                GetMutableTensor(out));
#endif
    }
  }
  SetDistProps(out, in.dims(), out_dist_attr);
}

}  // namespace phi::distributed
