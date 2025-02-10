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

#include <map>
#include <memory>
#include <unordered_map>

#include "paddle/fluid/platform/profiler/event_node.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/profiler/extra_info.h"

namespace paddle {
namespace platform {

struct DevicePythonNode {
  DevicePythonNode() = default;
  ~DevicePythonNode() {}
  // record name
  std::string name;
  // record type, one of TracerEventType
  TracerEventType type;
  // start timestamp of the record
  uint64_t start_ns;
  // end timestamp of the record
  uint64_t end_ns;
  // device id
  uint64_t device_id;
  // context id
  uint64_t context_id;
  // stream id
  uint64_t stream_id;
  // correlation id, used for correlating async activities happened on device
  uint32_t correlation_id;
  // The X-dimension block size for the kernel.
  uint32_t block_x;
  // The Y-dimension block size for the kernel.
  uint32_t block_y;
  // The Z-dimension grid size for the kernel.
  uint32_t block_z;
  // X-dimension of a grid.
  uint32_t grid_x;
  // Y-dimension of a grid.
  uint32_t grid_y;
  // Z-dimension of a grid.
  uint32_t grid_z;
  // dynamic + static
  uint64_t shared_memory;
  // The number of registers required for each thread executing the kernel.
  uint32_t registers_per_thread;
  float blocks_per_sm;
  float warps_per_sm;
  // theoretical achieved occupancy
  float occupancy;
  // The number of bytes transferred by the memory copy.
  uint64_t num_bytes;
  // the value being assigned to memory by the memory set.
  uint32_t value;
};

struct MemPythonNode {
  MemPythonNode() = default;
  ~MemPythonNode() {}

  // timestamp of the record
  uint64_t timestamp_ns;
  // memory addr of allocation or free
  uint64_t addr;
  // memory manipulation type
  TracerMemEventType type;
  // process id of the record
  uint64_t process_id;
  // thread id of the record
  uint64_t thread_id;
  // increase bytes after this manipulation, allocation for sign +, free for
  // sign -
  int64_t increase_bytes;
  // place
  std::string place;
  // current total allocated memory
  uint64_t current_allocated;
  // current total reserved memory
  uint64_t current_reserved;
  // peak  allocated memory
  uint64_t peak_allocated;
  // peak  reserved memory
  uint64_t peak_reserved;
};

struct HostPythonNode {
  HostPythonNode() = default;
  ~HostPythonNode();
  // record name
  std::string name;
  // record type, one of TracerEventType
  TracerEventType type;
  // start timestamp of the record
  uint64_t start_ns;
  // end timestamp of the record
  uint64_t end_ns;
  // process id of the record
  uint64_t process_id;
  // thread id of the record
  uint64_t thread_id;
  // correlation id, used for correlating async activities happened on device
  uint32_t correlation_id;
  // input shapes
  std::map<std::string, std::vector<std::vector<int64_t>>> input_shapes;
  std::map<std::string, std::vector<std::string>> dtypes;
  // call stack
  std::string callstack;
  // op attributes
  framework::AttributeMap attributes;
  // op id
  uint64_t op_id;
  // children node
  std::vector<HostPythonNode*> children_node_ptrs;
  // runtime node
  std::vector<HostPythonNode*> runtime_node_ptrs;
  // device node
  std::vector<DevicePythonNode*> device_node_ptrs;
  // mem node
  std::vector<MemPythonNode*> mem_node_ptrs;
};

class ProfilerResult {
 public:
  ProfilerResult() : tree_(nullptr) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  explicit ProfilerResult(
      std::unique_ptr<NodeTrees> tree,
      ExtraInfo extra_info,
      const std::map<uint32_t, gpuDeviceProp> device_property_map);
#endif
  explicit ProfilerResult(std::unique_ptr<NodeTrees> tree,
                          ExtraInfo extra_info);

  ~ProfilerResult();
  std::map<uint64_t, HostPythonNode*> GetData() {
    return thread_event_trees_map_;
  }
  std::unordered_map<std::string, std::string> GetExtraInfo() {
    return extra_info_.GetExtraInfo();
  }

  void Save(const std::string& file_name,
            const std::string format = std::string("json"));

  std::shared_ptr<NodeTrees> GetNodeTrees() { return tree_; }

  void SetVersion(const std::string& version) { version_ = version; }

  void SetSpanIndx(uint32_t span_indx) { span_indx_ = span_indx; }

  std::string GetVersion() { return version_; }
  uint32_t GetSpanIndx() { return span_indx_; }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::map<uint32_t, gpuDeviceProp> GetDeviceProperty() {
    return device_property_map_;
  }
#endif

 private:
  std::map<uint64_t, HostPythonNode*> thread_event_trees_map_;
  std::shared_ptr<NodeTrees> tree_;
  ExtraInfo extra_info_;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::map<uint32_t, gpuDeviceProp> device_property_map_;
#endif
  std::string version_;
  uint32_t span_indx_;
  HostPythonNode* CopyTree(HostTraceEventNode* root);
};

std::unique_ptr<ProfilerResult> LoadProfilerResult(std::string filename);

}  // namespace platform
}  // namespace paddle
