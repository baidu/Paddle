// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"

#include <set>
#include <thread>

#include "paddle/common/flags.h"
#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/utils/string/string_helper.h"

// FLAGS_force_sync_ops is used to finer control the op-sync in executor.
// The format is: "micro_batch_id, job_name, op_id, op_name | micro_batch_id,
// job_name, op_id, op_name | ...". Keep spaces to syncs all name/id. Example:
// 1. sync the p_recv op in the second backward-job of 1F1B scheduling:
// FLAGS_force_sync_ops="1, backward, , p_recv"
// 2. sync the full op with op_id=5: FLAGS_force_sync_ops=" , , 5, full"
// 3. sync all ops in the first default-job: FLAGS_force_sync_ops="0,default,,
// 4. sync all ops in the forward-job and backward-job: FLAGS_force_sync_ops=" ,
// forward, , | , backward, , , "
PHI_DEFINE_EXPORTED_string(force_sync_ops,
                           "",
                           "Pattern to force sync ops in executor.");

PD_DECLARE_bool(new_executor_serial_run);

namespace paddle::framework::interpreter {

static constexpr size_t kHostNumThreads = 4;
static constexpr size_t kDeviceNumThreads = 1;
static constexpr size_t kNumGcThreads = 1;

// By default, one interpretercore contains:
// 1-size thread pool for device kernel launch (or 0 for cpu execution),
// 1-size thread pool for host kernel launch (or more if the system contains
// enough processors).

// Note that the purpose of the config is to limit the total 'possible'
// threads introduced by interpretercore to avoid hurting performance.

inline std::tuple<int, int> GetThreadPoolConfig(const phi::Place& place,
                                                size_t op_num) {
  int num_device_threads = kDeviceNumThreads,
      num_host_threads = kHostNumThreads;

  int device_count = 0, processor_count = 0;
  if (phi::is_cpu_place(place)) {
    num_device_threads = 0;
    num_host_threads = 4;
  } else {
    processor_count = static_cast<int>(std::thread::hardware_concurrency());
    if (processor_count) {
      if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        device_count = phi::backends::gpu::GetGPUDeviceCount();
#endif
      }
      if (phi::is_xpu_place(place)) {
#if defined(PADDLE_WITH_XPU)
        device_count = phi::backends::xpu::GetXPUDeviceCount();
#endif
      }
      if (phi::is_ipu_place(place)) {
#if defined(PADDLE_WITH_IPU)
        device_count = platform::GetIPUDeviceCount();
#endif
      }
      if (phi::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
        device_count =
            phi::DeviceManager::GetDeviceCount(place.GetDeviceType());
#endif
      }

      // Tricky implementation.
      // In multi-card training, each card may set env like
      // CUDA_VISIBLE_DEVICE=0 In that case, device_count is set to 8.
      if (device_count == 1) {
        device_count = 8;  // in many case, the accelerator has 8 cards.
      }

      // We expect processor_count = 2 * (the possible total threads when doing
      // multi-card training), to make sure that the system will not slow down
      // because of too many threads. Here, 2 is experience value. Since each
      // device has one interpretercore, the possible total threads when doing
      // multi-card training = device_count * (the possible total threads in one
      // interpretercore).

      if (device_count) {
        auto num = processor_count / device_count / 2 -
                   (kNumGcThreads + num_device_threads);
        num_host_threads = static_cast<int>(
            num > 0 ? (num > kHostNumThreads ? kHostNumThreads : num) : 1);
      }
    }
  }

  // In serial run, only one 1-size thread pool is used
  if (FLAGS_new_executor_serial_run) {
    num_host_threads = 0;
    num_device_threads = 1;
  }

  VLOG(4) << "place:" << place << ", processor_count:" << processor_count
          << ", device_count:" << device_count
          << ", serial_run:" << FLAGS_new_executor_serial_run
          << ", num_host_threads:" << num_host_threads
          << ", num_device_threads:" << num_device_threads;

  return std::make_tuple(num_host_threads, num_device_threads);
}

void ExecutionConfig::AnalyzeThreadPoolConfig(const phi::Place& place,
                                              size_t op_num) {
  if (host_num_threads == 0 || device_num_threads == 0) {
    std::tie(host_num_threads, device_num_threads) =
        GetThreadPoolConfig(place, op_num);
  }
}

void ExecutionConfig::Log(int log_level) {
  std::stringstream log_str;
  log_str << "ExecutionConfig:\n"
          << "create_local_scope = " << create_local_scope << "\n"
          << "used_for_cinn = " << used_for_cinn << "\n"
          << "used_for_control_flow_op = " << used_for_control_flow_op << "\n"
          << "used_for_jit = " << used_for_jit << "\n"
          << "used_for_sot = " << used_for_sot << "\n"
          << "device_num_threads = " << device_num_threads << "\n"
          << "host_num_threads = " << host_num_threads << "\n";

  log_str << "force_root_scope_vars = [";
  for (const std::string& var : force_root_scope_vars) {
    log_str << var << " ";
  }
  log_str << "]\n";

  log_str << "jit_input_vars = [";
  for (const std::string& var : jit_input_vars) {
    log_str << var << " ";
  }
  log_str << "]\n";

  log_str << "skip_gc_vars = [";
  for (const std::string& var : skip_gc_vars) {
    log_str << var << " ";
  }
  log_str << "]\n";

  VLOG(log_level) << log_str.str();
}

std::set<std::pair<int, std::string>> GetForceSyncOps(
    int micro_batch_id, const std::string& job_name) {
  std::set<std::pair<int, std::string>> force_sync_ops;
  std::stringstream ss(paddle::string::erase_spaces(FLAGS_force_sync_ops));
  std::string item;

  while (std::getline(ss, item, '|')) {
    item += ",";  // The comma at the end of the string will be ignored in
                  // std::getline
    std::stringstream item_stream(item);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(item_stream, token, ',')) {
      VLOG(1) << "token: " << token;
      tokens.push_back(token);
    }

    PADDLE_ENFORCE_EQ(
        tokens.size(),
        4,
        phi::errors::InvalidArgument("Invalid force_sync_ops format: \"%s\", "
                                     "FLAGS_force_sync_ops=\"%s\"",
                                     item,
                                     FLAGS_force_sync_ops));

    int micro_batch_id_;
    if (tokens[0] == "") {
      micro_batch_id_ = -1;
    } else {
      micro_batch_id_ = std::stoi(tokens[0]);
    }
    if (micro_batch_id_ != micro_batch_id && micro_batch_id_ != -1) {
      continue;
    }

    if (tokens[1] != job_name && tokens[1] != "") {
      continue;
    }

    int op_id;
    if (tokens[2] == "") {
      op_id = -1;
    } else {
      op_id = std::stoi(tokens[2]);
    }
    std::string op_name = tokens[3];
    force_sync_ops.insert({op_id, op_name});
  }

  if (!force_sync_ops.empty()) {
    std::stringstream ss;
    ss << "job_name: " << job_name << ", micro_batch_id: " << micro_batch_id
       << ", force_sync_ops: ";
    for (auto& pair : force_sync_ops) {
      ss << "(" << pair.first << ", " << pair.second << ") ";
    }
    VLOG(6) << ss.str();
  }
  return force_sync_ops;
}

}  // namespace paddle::framework::interpreter
