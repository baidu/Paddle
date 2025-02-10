/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>

#include "paddle/fluid/platform/profiler/dump/nodetree.pb.h"
#include "paddle/fluid/platform/profiler/event_python.h"

namespace paddle {
namespace platform {

class DeserializationReader {
 public:
  explicit DeserializationReader(std::string filename);
  explicit DeserializationReader(const char* filename);
  ~DeserializationReader();
  std::unique_ptr<ProfilerResult> Parse();

 private:
  void OpenFile();
  DeviceTraceEventNode* RestoreDeviceTraceEventNode(
      const DeviceTraceEventNodeProto&);
  CudaRuntimeTraceEventNode* RestoreCudaRuntimeTraceEventNode(
      const CudaRuntimeTraceEventNodeProto&);
  HostTraceEventNode* RestoreHostTraceEventNode(const HostTraceEventNodeProto&);
  KernelEventInfo HandleKernelEventInfoProto(const DeviceTraceEventProto&);
  MemcpyEventInfo HandleMemcpyEventInfoProto(const DeviceTraceEventProto&);
  MemsetEventInfo HandleMemsetEventInfoProto(const DeviceTraceEventProto&);
  MemTraceEventNode* RestoreMemTraceEventNode(const MemTraceEventNodeProto&);
  OperatorSupplementEventNode* RestoreOperatorSupplementEventNode(
      const OperatorSupplementEventNodeProto&);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  gpuDeviceProp RestoreDeviceProperty(const DevicePropertyProto&);
#endif

  std::string filename_;
  std::ifstream input_file_stream_;
  NodeTreesProto* node_trees_proto_;
};

}  // namespace platform
}  // namespace paddle
