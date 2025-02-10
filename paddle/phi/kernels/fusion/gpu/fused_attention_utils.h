// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#endif
#include "paddle/common/errors.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace fusion {

template <typename T>
static void AllReduce(phi::DenseTensor &tensor,  // NOLINT
                      const int ring_id,
                      const phi::GPUContext &dev_ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  distributed::ProcessGroup *pg = nullptr;
  auto map = distributed::ProcessGroupMapFromGid::getInstance();
  if (map->has(ring_id)) {
    // Use ProcessGroup
    pg = map->get(ring_id);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "ring_id %d is not in ProcessGroupMap, please check related"
        "configurations and retry.",
        ring_id));
  }

  distributed::AllreduceOptions opts;
  opts.reduce_op = distributed::ReduceOp::SUM;
  auto task = pg->AllReduce(&tensor, tensor, opts, false, false);
  task->Wait();

#else
  PADDLE_THROW(common::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

template <typename T>
static void AllReduce(phi::DenseTensor &tensor,  // NOLINT
                      const int ring_id,
                      const int count UNUSED,
                      const phi::GPUContext &ctx) {
  AllReduce<T>(tensor, ring_id, ctx);
}

}  // namespace fusion
}  // namespace phi
