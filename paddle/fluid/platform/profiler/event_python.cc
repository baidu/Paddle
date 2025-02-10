/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/profiler/event_python.h"

#include <utility>

#include "paddle/fluid/platform/profiler/chrometracing_logger.h"
#include "paddle/fluid/platform/profiler/dump/deserialization_reader.h"
#include "paddle/fluid/platform/profiler/dump/serialization_logger.h"
#include "paddle/phi/core/platform/profiler/extra_info.h"

namespace paddle::platform {

HostPythonNode::~HostPythonNode() {
  // delete all runtime or device nodes and recursive delete children
  for (auto& children_node_ptr : children_node_ptrs) {
    delete children_node_ptr;
  }
  for (auto& runtime_node_ptr : runtime_node_ptrs) {
    delete runtime_node_ptr;
  }
  for (auto& device_node_ptr : device_node_ptrs) {
    delete device_node_ptr;
  }
  for (auto& mem_node_ptr : mem_node_ptrs) {
    delete mem_node_ptr;
  }
}

HostPythonNode* ProfilerResult::CopyTree(HostTraceEventNode* root) {
  // Copy and transfer EventNode in NodeTree to PythonNode
  if (root == nullptr) {
    return nullptr;
  }
  // copy HostTraceEventNode and its children
  HostPythonNode* host_python_node = new HostPythonNode();
  host_python_node->name = root->Name();
  host_python_node->type = root->Type();
  host_python_node->start_ns = root->StartNs();
  host_python_node->end_ns = root->EndNs();
  host_python_node->process_id = root->ProcessId();
  host_python_node->thread_id = root->ThreadId();
  for (auto child : root->GetChildren()) {
    host_python_node->children_node_ptrs.push_back(CopyTree(child));
  }
  // copy its CudaRuntimeTraceEventNode
  for (auto runtimenode : root->GetRuntimeTraceEventNodes()) {
    HostPythonNode* runtime_python_node = new HostPythonNode();
    runtime_python_node->name = runtimenode->Name();
    runtime_python_node->type = runtimenode->Type();
    runtime_python_node->start_ns = runtimenode->StartNs();
    runtime_python_node->end_ns = runtimenode->EndNs();
    runtime_python_node->process_id = runtimenode->ProcessId();
    runtime_python_node->thread_id = runtimenode->ThreadId();
    runtime_python_node->correlation_id = runtimenode->CorrelationId();
    host_python_node->runtime_node_ptrs.push_back(runtime_python_node);
    // copy DeviceTraceEventNode
    for (auto devicenode : runtimenode->GetDeviceTraceEventNodes()) {
      DevicePythonNode* device_python_node = new DevicePythonNode();
      device_python_node->name = devicenode->Name();
      device_python_node->type = devicenode->Type();
      device_python_node->start_ns = devicenode->StartNs();
      device_python_node->end_ns = devicenode->EndNs();
      device_python_node->device_id = devicenode->DeviceId();
      device_python_node->context_id = devicenode->ContextId();
      device_python_node->stream_id = devicenode->StreamId();
      device_python_node->correlation_id = devicenode->CorrelationId();
      if (device_python_node->type == TracerEventType::Kernel) {
        KernelEventInfo kernel_info = devicenode->KernelInfo();
        device_python_node->block_x = kernel_info.block_x;
        device_python_node->block_y = kernel_info.block_y;
        device_python_node->block_z = kernel_info.block_z;
        device_python_node->grid_x = kernel_info.grid_x;
        device_python_node->grid_y = kernel_info.grid_y;
        device_python_node->grid_z = kernel_info.grid_z;
        device_python_node->shared_memory = kernel_info.dynamic_shared_memory +
                                            kernel_info.static_shared_memory;
        device_python_node->registers_per_thread =
            kernel_info.registers_per_thread;
        device_python_node->blocks_per_sm = kernel_info.blocks_per_sm;
        device_python_node->warps_per_sm = kernel_info.warps_per_sm;
        device_python_node->occupancy = kernel_info.occupancy;
      } else if (device_python_node->type == TracerEventType::Memcpy) {
        MemcpyEventInfo memcpy_info = devicenode->MemcpyInfo();
        device_python_node->num_bytes = memcpy_info.num_bytes;
      } else if (device_python_node->type == TracerEventType::Memset) {
        MemsetEventInfo memset_info = devicenode->MemsetInfo();
        device_python_node->num_bytes = memset_info.num_bytes;
        device_python_node->value = memset_info.value;
      }
      runtime_python_node->device_node_ptrs.push_back(device_python_node);
    }
  }
  // copy MemTraceEventNode
  for (auto memnode : root->GetMemTraceEventNodes()) {
    MemPythonNode* mem_python_node = new MemPythonNode();
    mem_python_node->timestamp_ns = memnode->TimeStampNs();
    mem_python_node->addr = memnode->Addr();
    mem_python_node->type = memnode->Type();
    mem_python_node->process_id = memnode->ProcessId();
    mem_python_node->thread_id = memnode->ThreadId();
    mem_python_node->increase_bytes = memnode->IncreaseBytes();
    mem_python_node->place = memnode->Place();
    mem_python_node->current_allocated = memnode->CurrentAllocated();
    mem_python_node->current_reserved = memnode->CurrentReserved();
    mem_python_node->peak_allocated = memnode->PeakAllocated();
    mem_python_node->peak_reserved = memnode->PeakReserved();
    host_python_node->mem_node_ptrs.push_back(mem_python_node);
  }
  // copy OperatorSupplementEventNode's information if exists
  OperatorSupplementEventNode* op_supplement_node =
      root->GetOperatorSupplementEventNode();
  if (op_supplement_node != nullptr) {
    host_python_node->input_shapes = op_supplement_node->InputShapes();
    host_python_node->dtypes = op_supplement_node->Dtypes();
    host_python_node->callstack = op_supplement_node->CallStack();
    host_python_node->attributes = op_supplement_node->Attributes();
    host_python_node->op_id = op_supplement_node->OpId();
  }
  return host_python_node;
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
ProfilerResult::ProfilerResult(
    std::unique_ptr<NodeTrees> tree,
    ExtraInfo extra_info,
    const std::map<uint32_t, gpuDeviceProp> device_property_map)
    : tree_(tree.release()),
<<<<<<< HEAD
      extra_info_(std::move(extra_info)),
      device_property_map_(device_property_map) {
=======
      extra_info_(extra_info),
      device_property_map_(device_property_map),
      span_indx_(0) {
>>>>>>> 2b4a472bb38fae94a9eb0c5ddc3e16f316f4c5df
  if (tree_ != nullptr) {
    std::map<uint64_t, HostTraceEventNode*> nodetrees = tree_->GetNodeTrees();
    for (auto& nodetree : nodetrees) {
      thread_event_trees_map_[nodetree.first] = CopyTree(nodetree.second);
    }
  }
}
#endif

ProfilerResult::ProfilerResult(std::unique_ptr<NodeTrees> tree,
<<<<<<< HEAD
                               ExtraInfo extra_info)
    : tree_(tree.release()), extra_info_(std::move(extra_info)) {
=======
                               const ExtraInfo& extra_info)
    : tree_(tree.release()), extra_info_(extra_info), span_indx_(0) {
>>>>>>> 2b4a472bb38fae94a9eb0c5ddc3e16f316f4c5df
  if (tree_ != nullptr) {
    std::map<uint64_t, HostTraceEventNode*> nodetrees = tree_->GetNodeTrees();
    for (auto& nodetree : nodetrees) {
      thread_event_trees_map_[nodetree.first] = CopyTree(nodetree.second);
    }
  }
}

ProfilerResult::~ProfilerResult() {
  // delete all root nodes
  for (auto& item : thread_event_trees_map_) {
    delete item.second;
  }
}

void ProfilerResult::Save(const std::string& file_name,
                          const std::string format) {
  if (format == std::string("json")) {
    ChromeTracingLogger logger(file_name);
    logger.LogMetaInfo(version_, span_indx_);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    logger.LogDeviceProperty(device_property_map_);
#endif
    tree_->LogMe(&logger);
    logger.LogExtraInfo(GetExtraInfo());
  } else if (format == std::string("pb")) {
    SerializationLogger logger(file_name);
    logger.LogMetaInfo(version_, span_indx_);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    logger.LogDeviceProperty(device_property_map_);
#endif
    tree_->LogMe(&logger);
    logger.LogExtraInfo(GetExtraInfo());
  }
  return;
}

std::unique_ptr<ProfilerResult> LoadProfilerResult(std::string filename) {
  DeserializationReader reader(filename);
  std::unique_ptr<ProfilerResult> result = reader.Parse();
  return result;
}

}  // namespace paddle::platform
