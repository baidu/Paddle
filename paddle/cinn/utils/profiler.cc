// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/utils/profiler.h"

#include "paddle/common/flags.h"

#ifdef CINN_WITH_NVTX
#include <nvToolsExt.h>
#endif
#ifdef CINN_WITH_CUDA
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "paddle/cinn/backends/cuda_util.h"
#endif
#include <chrono>

PD_DECLARE_int32(cinn_profiler_state);

namespace cinn {
namespace utils {

ProfilerState ProfilerHelper::g_state = ProfilerState::kDisabled;

void ProfilerHelper::UpdateState() {
  if (FLAGS_cinn_profiler_state < 0) return;

  switch (FLAGS_cinn_profiler_state) {
    case 0:
      g_state = ProfilerState::kDisabled;
      break;
    case 1:
      g_state = ProfilerState::kCPU;
      break;
    case 2:
      g_state = ProfilerState::kCUDA;
      break;
    case 3:
      g_state = ProfilerState::kAll;
      break;
    default:
      LOG(WARNING) << "Unsupported FLAGS_cinn_profiler_state = "
                   << FLAGS_cinn_profiler_state << ", and will do nothing.";
  }
}

RecordEvent::RecordEvent(const std::string& name, EventType type) {
  if (!ProfilerHelper::IsEnable()) return;

  if (ProfilerHelper::IsEnableCPU()) {
    call_back_ = [this,
                  tik = std::chrono::steady_clock::now(),
                  annotation = std::move(name),
                  type]() {
      auto tok = std::chrono::steady_clock::now();
      std::chrono::duration<double> duration = (tok - tik) * 1e3;  // ms
      HostEventRecorder::GetInstance().RecordEvent(
          annotation, duration.count(), type);
    };
  }

  if (ProfilerHelper::IsEnableCUDA()) {
    ProfilerRangePush(name);
  }
}

void RecordEvent::End() {
  if (!ProfilerHelper::IsEnable()) return;

  if (ProfilerHelper::IsEnableCPU() && call_back_ != nullptr) {
    call_back_();
  }

  if (ProfilerHelper::IsEnableCUDA()) {
    ProfilerRangePop();
  }
}

void SynchronizeAllDevice() {
#ifdef CINN_WITH_CUDA
  int current_device_id;
  CUDA_CALL(cudaGetDevice(&current_device_id));
  int count;
  CUDA_CALL(cudaGetDeviceCount(&count));
  for (int i = 0; i < count; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaDeviceSynchronize());
  }
  CUDA_CALL(cudaSetDevice(current_device_id));
#endif
}

void ProfilerStart() {
#ifdef CINN_WITH_CUDA
  CUDA_CALL(cudaProfilerStart());
  SynchronizeAllDevice();
#endif
}

void ProfilerStop() {
#ifdef CINN_WITH_CUDA
  CUDA_CALL(cudaProfilerStop());
#endif
}

void ProfilerRangePush(const std::string& name) {
#ifdef CINN_WITH_NVTX
  nvtxRangePushA(name.c_str());
#endif
}

void ProfilerRangePop() {
#ifdef CINN_WITH_NVTX
  nvtxRangePop();
#endif
}

}  // namespace utils
}  // namespace cinn
