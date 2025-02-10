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

// old op include, fluid should be removed
#ifdef PADDLE_WITH_HIP
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <vector>
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/impl/softmax_kernel_impl.h"
#include "paddle/phi/kernels/margin_cross_entropy_grad_kernel.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T, typename Context>
void GetClassInterval(const gpuStream_t& stream,
                      const phi::Place& place,
                      const Context& dev_ctx,
                      const int rid,
                      const int rank,
                      const int nranks,
                      const int D,
                      DenseTensor* class_interval) {
  std::vector<int> shard_dim_vec(nranks + 1, 0);
  shard_dim_vec[rank + 1] = D;
  if (nranks <= 1) {
    phi::TensorFromVector(shard_dim_vec, dev_ctx, class_interval);
    return;
  }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  DenseTensor num_classes_per_device;
  phi::TensorFromVector(shard_dim_vec, dev_ctx, &num_classes_per_device);
  int* num_classes_per_device_ptr = num_classes_per_device.data<int>();

  auto map = distributed::ProcessGroupMapFromGid::getInstance();
  if (map->has(rid)) {
    // Use ProcessGroup
    distributed::ProcessGroup* pg = map->get(rid);
    std::vector<phi::DenseTensor> in_tensor;
    std::vector<phi::DenseTensor> out_tensor;
    in_tensor.push_back(num_classes_per_device);
    out_tensor.push_back(num_classes_per_device);

    distributed::AllreduceOptions opts;
    opts.reduce_op = distributed::ReduceOp::SUM;
    auto task = pg->AllReduce(in_tensor, out_tensor, opts);
    task->Wait();
  } else {
    phi::distributed::NCCLCommContext* comm_ctx =
        static_cast<phi::distributed::NCCLCommContext*>(
            dev_ctx.GetCommContext());
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      common::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));

    // use global calculate stream
    const auto calcu_stream =
        static_cast<GPUContext*>(phi::DeviceContextPool::Instance().Get(place))
            ->stream();
    comm_ctx->AllReduce(
        &num_classes_per_device, num_classes_per_device, ncclSum, calcu_stream);
  }

  class_interval->Resize({nranks + 1});
  auto class_interval_ptr = dev_ctx.template Alloc<int>(class_interval);
  size_t cub_temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum<int*, int*>(
      nullptr, cub_temp_storage_bytes, nullptr, nullptr, nranks + 1, stream);
  auto cub_temp_storage =
      phi::memory_utils::Alloc(place, cub_temp_storage_bytes);
  cub::DeviceScan::InclusiveSum<int*, int*>(cub_temp_storage->ptr(),
                                            cub_temp_storage_bytes,
                                            num_classes_per_device_ptr,
                                            class_interval_ptr,
                                            nranks + 1,
                                            stream);
  return;
#endif
}

}  // namespace phi
