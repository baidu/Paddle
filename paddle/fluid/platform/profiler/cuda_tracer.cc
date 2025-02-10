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

#include "paddle/fluid/platform/profiler/cuda_tracer.h"

#include <string>
#include <unordered_map>

#include "glog/logging.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue_utils.h"
#include "paddle/fluid/platform/profiler/cupti_data_process.h"
#include "paddle/phi/core/os_info.h"

#define CUPTI_CALL(call)                                                     \
  do {                                                                       \
    CUptiResult _status = call;                                              \
    if (_status != CUPTI_SUCCESS) {                                          \
      const char* errstr;                                                    \
      phi::dynload::cuptiGetResultString(_status, &errstr);                  \
      LOG(ERROR) << "Function " << #call << " failed with error " << errstr; \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)

namespace paddle::platform::details {
std::unordered_map<uint32_t, uint64_t> CreateThreadIdMapping() {
  std::unordered_map<uint32_t, uint64_t> mapping;
  std::unordered_map<uint64_t, phi::ThreadId> ids = phi::GetAllThreadIds();
  for (const auto& id : ids) {
    mapping[id.second.cupti_tid] = id.second.sys_tid;
  }
  return mapping;
}
}  // namespace paddle::platform::details
namespace paddle::platform {

CudaTracer::CudaTracer() = default;

void CudaTracer::PrepareTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::UNINITED || state_ == TracerState::STOPPED,
      true,
      common::errors::PreconditionNotMet("Tracer must be UNINITED"));
  EnableCuptiActivity();
  state_ = TracerState::READY;
}

void CudaTracer::StartTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::READY,
      true,
      common::errors::PreconditionNotMet("Tracer must be READY or STOPPED"));
  ConsumeBuffers();
  tracing_start_ns_ = phi::PosixInNsec();
  state_ = TracerState::STARTED;
}

void CudaTracer::StopTracing() {
  PADDLE_ENFORCE_EQ(
      state_,
      TracerState::STARTED,
      common::errors::PreconditionNotMet("Tracer must be STARTED"));
  DisableCuptiActivity();
  state_ = TracerState::STOPPED;
}

void CudaTracer::CollectTraceData(TraceEventCollector* collector) {
  PADDLE_ENFORCE_EQ(
      state_,
      TracerState::STOPPED,
      common::errors::PreconditionNotMet("Tracer must be STOPPED"));
  ProcessCuptiActivity(collector);
}

int CudaTracer::ProcessCuptiActivity(TraceEventCollector* collector) {
  int record_cnt = 0;
#ifdef PADDLE_WITH_CUPTI
  CUPTI_CALL(
      phi::dynload::cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  auto mapping = details::CreateThreadIdMapping();
  std::vector<ActivityBuffer> buffers = ConsumeBuffers();
  for (auto& buffer : buffers) {
    if (buffer.addr == nullptr || buffer.valid_size == 0) {
      continue;
    }

    CUpti_Activity* record = nullptr;
    while (true) {
      CUptiResult status = phi::dynload::cuptiActivityGetNextRecord(
          buffer.addr, buffer.valid_size, &record);
      if (status == CUPTI_SUCCESS) {
        details::ProcessCuptiActivityRecord(
            record, tracing_start_ns_, mapping, collector);
        ++record_cnt;
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        CUPTI_CALL(status);
      }
    }

    ReleaseBuffer(buffer.addr);
  }
#endif
  return record_cnt;
}

void CudaTracer::EnableCuptiActivity() {
#ifdef PADDLE_WITH_CUPTI
  CUPTI_CALL(phi::dynload::cuptiActivityRegisterCallbacks(
      BufferRequestedCallback, BufferCompletedCallback));

  CUPTI_CALL(phi::dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(
      phi::dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(phi::dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(phi::dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(phi::dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  VLOG(3) << "enable cupti activity";
#endif
}

void CudaTracer::DisableCuptiActivity() {
#ifdef PADDLE_WITH_CUPTI
  CUPTI_CALL(phi::dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(phi::dynload::cuptiActivityDisable(
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(phi::dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(phi::dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(phi::dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  VLOG(3) << "disable cupti activity";
#endif
}

#ifdef PADDLE_WITH_CUPTI
void CUPTIAPI CudaTracer::BufferRequestedCallback(uint8_t** buffer,
                                                  size_t* size,
                                                  size_t* max_num_records) {
  GetInstance().AllocateBuffer(buffer, size);
  *max_num_records = 0;
}

void CUPTIAPI CudaTracer::BufferCompletedCallback(CUcontext ctx,
                                                  uint32_t stream_id,
                                                  uint8_t* buffer,
                                                  size_t size,
                                                  size_t valid_size) {
  GetInstance().ProduceBuffer(buffer, valid_size);
  size_t dropped = 0;
  CUPTI_CALL(phi::dynload::cuptiActivityGetNumDroppedRecords(
      ctx, stream_id, &dropped));
  if (dropped != 0) {
    LOG(WARNING) << "Stream " << stream_id << " Dropped " << dropped
                 << " activity records";
  }
}
#endif

void CudaTracer::AllocateBuffer(uint8_t** buffer, size_t* size) {
  constexpr size_t kBufSize = 1 << 23;  // 8 MB
  constexpr size_t kBufAlign = 8;       // 8 B
  *buffer = reinterpret_cast<uint8_t*>(
      paddle::framework::AlignedMalloc(kBufSize, kBufAlign));
  *size = kBufSize;
}

void CudaTracer::ProduceBuffer(uint8_t* buffer, size_t valid_size) {
  std::lock_guard<std::mutex> guard(activity_buffer_lock_);
  activity_buffers_.emplace_back(buffer, valid_size);
}

std::vector<CudaTracer::ActivityBuffer> CudaTracer::ConsumeBuffers() {
  std::vector<ActivityBuffer> buffers;
  {
    std::lock_guard<std::mutex> guard(activity_buffer_lock_);
    buffers.swap(activity_buffers_);
  }
  return buffers;
}

void CudaTracer::ReleaseBuffer(uint8_t* buffer) {
  paddle::framework::AlignedFree(buffer);
}

}  // namespace paddle::platform
