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

#include "paddle/cinn/runtime/hip/hip_util.h"
#include <glog/logging.h>
#include "paddle/cinn/utils/profiler.h"

namespace cinn {
namespace runtime {
namespace hip {

void cinn_call_hip_kernel(void *kernel_fn,
                          void *v_args,
                          int num_args,
                          int grid_x,
                          int grid_y,
                          int grid_z,
                          int block_x,
                          int block_y,
                          int block_z,
                          int shared_memory_bytes,
                          void *stream) {
  int current_device_id;
  hipGetDevice(&current_device_id);
  VLOG(3) << "cinn_call_hip_kernel, grid_dim={" << grid_x << ", " << grid_y
          << ", " << grid_z << "}, block_dim={" << block_x << ", " << block_y
          << ", " << block_z << "}, num_args=" << num_args
          << ", shared_memory_bytes=" << shared_memory_bytes
          << ", stream=" << stream << ", kernel_fn=" << kernel_fn
          << " in device" << current_device_id;
  std::vector<void *> kernel_args;
  {
    cinn::utils::RecordEvent record_run("prepare_args",
                                        cinn::utils::EventType::kInstruction);
    kernel_args.reserve(num_args);
    cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
    for (int idx = 0; idx < num_args; ++idx) {
      if (args[idx].type_code() == ::cinn_type_code<cinn_buffer_t *>()) {
        kernel_args.emplace_back(
            &((cinn_buffer_t *)(args[idx]))->memory);  // NOLINT
      } else {
        kernel_args.emplace_back(args[idx].data_addr());
      }
    }
  }

  {
    cinn::utils::RecordEvent record_run("hipLaunchKernel",
                                        cinn::utils::EventType::kInstruction);
    HIP_DRIVER_CHECK(
        hipModuleLaunchKernel(static_cast<hipFunction_t>(kernel_fn),
                              grid_x,
                              grid_y,
                              grid_z,
                              block_x,
                              block_y,
                              block_z,
                              shared_memory_bytes,
                              static_cast<hipStream_t>(stream),
                              kernel_args.data(),
                              nullptr))
  }
}

void infer_shape_set_value(int row, int col, int64_t value, int64_t **v) {
  v[row][col] = value;
}

int64_t cinn_get_value_in_hip_kernel_args(void *v_args, int idx) {
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  return args[idx].operator int64_t();
}

void *cinn_get_item_in_hip_kernel_args(void *v_args, int idx) {
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  return static_cast<void *>(&args[idx]);
}

}  // namespace hip
}  // namespace runtime
}  // namespace cinn
