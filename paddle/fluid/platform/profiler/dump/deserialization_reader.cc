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
#include "paddle/fluid/platform/profiler/dump/deserialization_reader.h"

#include <cstring>
#include <utility>

#include "paddle/phi/core/platform/profiler/extra_info.h"

namespace paddle::platform {

DeserializationReader::DeserializationReader(std::string filename)
    : filename_(std::move(filename)) {
  OpenFile();
  node_trees_proto_ = new NodeTreesProto();
}

DeserializationReader::DeserializationReader(const char* filename)
    : filename_(filename) {
  OpenFile();
  node_trees_proto_ = new NodeTreesProto();
}

void DeserializationReader::OpenFile() {
  input_file_stream_.open(filename_, std::ifstream::in | std::ifstream::binary);
  if (!input_file_stream_) {
    VLOG(2) << "Unable to open file for writing profiling data." << std::endl;
  } else {
    VLOG(0) << "Read profiling data from " << filename_ << std::endl;
  }
}

std::unique_ptr<ProfilerResult> DeserializationReader::Parse() {
  if (!node_trees_proto_->ParseFromIstream(&input_file_stream_)) {
    VLOG(2) << "Unable to load node trees in protobuf." << std::endl;
    return nullptr;
  }
  // restore extra info
  ExtraInfo extra_info;
  for (auto indx = 0; indx < node_trees_proto_->extra_info_size(); indx++) {
    ExtraInfoMap extra_info_map = node_trees_proto_->extra_info(indx);
    extra_info.AddExtraInfo(extra_info_map.key(),
                            std::string("%s"),
                            extra_info_map.value().c_str());
  }

  // restore NodeTrees
  std::map<uint64_t, HostTraceEventNode*> thread_event_trees_map;
  for (int node_tree_index = 0;
       node_tree_index < node_trees_proto_->thread_trees_size();
       node_tree_index++) {
    // handle one thread tree
    std::map<int64_t, HostTraceEventNode*> index_node_map;
    std::map<int64_t, int64_t> child_parent_map;
    const ThreadNodeTreeProto& thread_node_tree_proto =
        node_trees_proto_->thread_trees(node_tree_index);
    uint64_t current_threadid = thread_node_tree_proto.thread_id();
    for (int host_node_index = 0;
         host_node_index < thread_node_tree_proto.host_nodes_size();
         host_node_index++) {
      // handle host node
      const HostTraceEventNodeProto& host_node_proto =
          thread_node_tree_proto.host_nodes(host_node_index);
      HostTraceEventNode* host_node =
          RestoreHostTraceEventNode(host_node_proto);
      index_node_map[host_node_proto.id()] = host_node;
      child_parent_map[host_node_proto.id()] = host_node_proto.parentid();
      // handle runtime node
      for (int runtime_node_index = 0;
           runtime_node_index < host_node_proto.runtime_nodes_size();
           runtime_node_index++) {
        const CudaRuntimeTraceEventNodeProto& runtime_node_proto =
            host_node_proto.runtime_nodes(runtime_node_index);
        CudaRuntimeTraceEventNode* runtime_node =
            RestoreCudaRuntimeTraceEventNode(runtime_node_proto);
        host_node->AddCudaRuntimeNode(runtime_node);  // insert into host_node
        // handle device node
        for (int device_node_index = 0;
             device_node_index < runtime_node_proto.device_nodes_size();
             device_node_index++) {
          const DeviceTraceEventNodeProto& device_node_proto =
              runtime_node_proto.device_nodes(device_node_index);
          DeviceTraceEventNode* device_node =
              RestoreDeviceTraceEventNode(device_node_proto);
          runtime_node->AddDeviceTraceEventNode(
              device_node);  // insert into runtime_node
        }
      }
      // handle mem node
      for (int mem_node_index = 0;
           mem_node_index < host_node_proto.mem_nodes_size();
           mem_node_index++) {
        const MemTraceEventNodeProto& mem_node_proto =
            host_node_proto.mem_nodes(mem_node_index);
        MemTraceEventNode* mem_node = RestoreMemTraceEventNode(mem_node_proto);
        host_node->AddMemNode(mem_node);
      }
      // handle op supplement node
      for (int op_supplement_node_index = 0;
           op_supplement_node_index <
           host_node_proto.op_supplement_nodes_size();
           op_supplement_node_index++) {
        const OperatorSupplementEventNodeProto& op_supplement_node_proto =
            host_node_proto.op_supplement_nodes(op_supplement_node_index);
        OperatorSupplementEventNode* op_supplement_node =
            RestoreOperatorSupplementEventNode(op_supplement_node_proto);
        host_node->SetOperatorSupplementNode(op_supplement_node);
      }
    }
    // restore parent-child relationship
    for (auto& map_item : child_parent_map) {
      if (map_item.second != -1) {  // not root node
        index_node_map[map_item.second]->AddChild(
            index_node_map[map_item.first]);
      } else {
        thread_event_trees_map[current_threadid] =
            index_node_map[map_item.first];  // root node
      }
    }
  }
  // restore NodeTrees object
  std::unique_ptr<NodeTrees> tree(new NodeTrees(thread_event_trees_map));
// restore gpuDeviceProp
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::map<uint32_t, gpuDeviceProp> device_property_map;
  for (auto indx = 0; indx < node_trees_proto_->device_property_size();
       indx++) {
    const DevicePropertyProto& device_property_proto =
        node_trees_proto_->device_property(indx);
    device_property_map[device_property_proto.id()] =
        RestoreDeviceProperty(device_property_proto);
  }
  ProfilerResult* profiler_result_ptr =
      new ProfilerResult(std::move(tree), extra_info, device_property_map);
#else
  ProfilerResult* profiler_result_ptr =
      new ProfilerResult(std::move(tree), extra_info);
#endif
  // restore version and span indx
  profiler_result_ptr->SetVersion(node_trees_proto_->version());
  profiler_result_ptr->SetSpanIndx(node_trees_proto_->span_indx());
  return std::unique_ptr<ProfilerResult>(profiler_result_ptr);
}

DeserializationReader::~DeserializationReader() {  // NOLINT
  delete node_trees_proto_;
  input_file_stream_.close();
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
gpuDeviceProp DeserializationReader::RestoreDeviceProperty(
    const DevicePropertyProto& device_property_proto) {
  gpuDeviceProp device_property;
  strncpy(device_property.name,
          device_property_proto.name().c_str(),
          device_property_proto.name().length() + 1);
  device_property.totalGlobalMem = device_property_proto.total_global_memory();
  device_property.major = device_property_proto.compute_major();  // NOLINT
  device_property.minor = device_property_proto.compute_minor();  // NOLINT
  device_property.multiProcessorCount =
      device_property_proto.sm_count();  // NOLINT
#if defined(PADDLE_WITH_CUDA)
  device_property.maxThreadsPerBlock =
      device_property_proto.max_threads_per_block();  // NOLINT
  device_property.maxThreadsPerMultiProcessor =
      device_property_proto.max_threads_per_multiprocessor();  // NOLINT
  device_property.regsPerBlock =
      device_property_proto.regs_per_block();  // NOLINT
  device_property.regsPerMultiprocessor =
      device_property_proto.regs_per_multiprocessor();           // NOLINT
  device_property.warpSize = device_property_proto.warp_size();  // NOLINT
  device_property.sharedMemPerBlock =
      device_property_proto.shared_memory_per_block();
  device_property.sharedMemPerMultiprocessor =
      device_property_proto.shared_memory_per_multiprocessor();
  device_property.sharedMemPerBlockOptin =
      device_property_proto.shared_memory_per_block_optin();
#endif
  return device_property;
}
#endif

DeviceTraceEventNode* DeserializationReader::RestoreDeviceTraceEventNode(
    const DeviceTraceEventNodeProto& device_node_proto) {
  const DeviceTraceEventProto& device_event_proto =
      device_node_proto.device_event();
  DeviceTraceEvent device_event;
  device_event.name = device_event_proto.name();
  device_event.type = static_cast<TracerEventType>(device_event_proto.type());
  device_event.start_ns = device_event_proto.start_ns();
  device_event.end_ns = device_event_proto.end_ns();
  device_event.device_id = device_event_proto.device_id();
  device_event.context_id = device_event_proto.context_id();
  device_event.stream_id = device_event_proto.stream_id();
  device_event.correlation_id = device_event_proto.correlation_id();
  switch (device_event.type) {
    case TracerEventType::Kernel:
      device_event.kernel_info = HandleKernelEventInfoProto(device_event_proto);
      break;

    case TracerEventType::Memcpy:
      device_event.memcpy_info = HandleMemcpyEventInfoProto(device_event_proto);
      break;

    case TracerEventType::Memset:
      device_event.memset_info = HandleMemsetEventInfoProto(device_event_proto);
      break;
    default:
      break;
  }
  return new DeviceTraceEventNode(device_event);
}

CudaRuntimeTraceEventNode*
DeserializationReader::RestoreCudaRuntimeTraceEventNode(
    const CudaRuntimeTraceEventNodeProto& runtime_node_proto) {
  const CudaRuntimeTraceEventProto& runtime_event_proto =
      runtime_node_proto.runtime_trace_event();
  RuntimeTraceEvent runtime_event;
  runtime_event.name = runtime_event_proto.name();
  runtime_event.start_ns = runtime_event_proto.start_ns();
  runtime_event.end_ns = runtime_event_proto.end_ns();
  runtime_event.process_id = runtime_event_proto.process_id();
  runtime_event.thread_id = runtime_event_proto.thread_id();
  runtime_event.correlation_id = runtime_event_proto.correlation_id();
  runtime_event.callback_id = runtime_event_proto.callback_id();
  return new CudaRuntimeTraceEventNode(runtime_event);
}

HostTraceEventNode* DeserializationReader::RestoreHostTraceEventNode(
    const HostTraceEventNodeProto& host_node_proto) {
  const HostTraceEventProto& host_event_proto =
      host_node_proto.host_trace_event();
  HostTraceEvent host_event;
  host_event.name = host_event_proto.name();
  host_event.type = static_cast<TracerEventType>(host_event_proto.type());
  host_event.start_ns = host_event_proto.start_ns();
  host_event.end_ns = host_event_proto.end_ns();
  host_event.process_id = host_event_proto.process_id();
  host_event.thread_id = host_event_proto.thread_id();
  return new HostTraceEventNode(host_event);
}

MemTraceEventNode* DeserializationReader::RestoreMemTraceEventNode(
    const MemTraceEventNodeProto& mem_node_proto) {
  const MemTraceEventProto& mem_event_proto = mem_node_proto.mem_event();
  MemTraceEvent mem_event;
  mem_event.timestamp_ns = mem_event_proto.timestamp_ns();
  mem_event.addr = mem_event_proto.addr();
  mem_event.type = static_cast<TracerMemEventType>(mem_event_proto.type());
  mem_event.process_id = mem_event_proto.process_id();
  mem_event.thread_id = mem_event_proto.thread_id();
  mem_event.increase_bytes = mem_event_proto.increase_bytes();
  mem_event.place = mem_event_proto.place();
  mem_event.current_allocated = mem_event_proto.current_allocated();
  mem_event.current_reserved = mem_event_proto.current_reserved();
  mem_event.peak_allocated = mem_event_proto.peak_allocated();
  mem_event.peak_reserved = mem_event_proto.peak_reserved();
  return new MemTraceEventNode(mem_event);
}

OperatorSupplementEventNode*
DeserializationReader::RestoreOperatorSupplementEventNode(
    const OperatorSupplementEventNodeProto& op_supplement_node_proto) {
  const OperatorSupplementEventProto& op_supplement_event_proto =
      op_supplement_node_proto.op_supplement_event();
  OperatorSupplementEvent op_supplement_event;
  op_supplement_event.timestamp_ns = op_supplement_event_proto.timestamp_ns();
  op_supplement_event.op_type = op_supplement_event_proto.op_type();
  op_supplement_event.callstack = op_supplement_event_proto.callstack();
  op_supplement_event.op_id = op_supplement_event_proto.op_id();
  op_supplement_event.process_id = op_supplement_event_proto.process_id();
  op_supplement_event.thread_id = op_supplement_event_proto.thread_id();
  std::map<std::string, std::vector<std::vector<int64_t>>> input_shapes;
  std::map<std::string, std::vector<std::string>> dtypes;
  auto input_shape_proto = op_supplement_event_proto.input_shapes();
  for (int i = 0; i < input_shape_proto.key_size(); i++) {
    auto input_shape_vec = input_shapes[input_shape_proto.key(i)];
    auto shape_vectors_proto = input_shape_proto.shape_vecs(i);
    for (int j = 0; j < shape_vectors_proto.shapes_size(); j++) {
      auto shape_vector_proto = shape_vectors_proto.shapes(j);
      std::vector<int64_t> shape;
      for (int k = 0; k < shape_vector_proto.size_size(); k++) {
        shape.push_back(shape_vector_proto.size(k));  // NOLINT
      }
      input_shape_vec.push_back(shape);
    }
  }
  op_supplement_event.input_shapes = input_shapes;
  auto dtype_proto = op_supplement_event_proto.dtypes();
  for (int i = 0; i < dtype_proto.key_size(); i++) {
    auto dtype_vec = dtypes[dtype_proto.key(i)];
    auto dtype_vec_proto = dtype_proto.dtype_vecs(i);
    for (int j = 0; j < dtype_vec_proto.dtype_size(); j++) {
      auto dtype_string = dtype_vec_proto.dtype(j);
      dtype_vec.push_back(dtype_string);
    }
  }
  op_supplement_event.dtypes = dtypes;
  return new OperatorSupplementEventNode(op_supplement_event);
}

KernelEventInfo DeserializationReader::HandleKernelEventInfoProto(
    const DeviceTraceEventProto& device_event_proto) {
  const KernelEventInfoProto& kernel_info_proto =
      device_event_proto.kernel_info();
  KernelEventInfo kernel_info;
  kernel_info.block_x = kernel_info_proto.block_x();
  kernel_info.block_y = kernel_info_proto.block_y();
  kernel_info.block_z = kernel_info_proto.block_z();
  kernel_info.grid_x = kernel_info_proto.grid_x();
  kernel_info.grid_y = kernel_info_proto.grid_y();
  kernel_info.grid_z = kernel_info_proto.grid_z();
  kernel_info.dynamic_shared_memory = kernel_info_proto.dynamic_shared_memory();
  kernel_info.static_shared_memory = kernel_info_proto.static_shared_memory();
  kernel_info.registers_per_thread = kernel_info_proto.registers_per_thread();
  kernel_info.local_memory_per_thread =
      kernel_info_proto.local_memory_per_thread();
  kernel_info.local_memory_total = kernel_info_proto.local_memory_total();
  kernel_info.queued = kernel_info_proto.queued();
  kernel_info.submitted = kernel_info_proto.submitted();
  kernel_info.completed = kernel_info_proto.completed();
  // version 1.0.2
  kernel_info.blocks_per_sm = kernel_info_proto.blocks_per_sm();
  kernel_info.warps_per_sm = kernel_info_proto.warps_per_sm();
  kernel_info.occupancy = kernel_info_proto.occupancy();
  return kernel_info;
}

MemcpyEventInfo DeserializationReader::HandleMemcpyEventInfoProto(
    const DeviceTraceEventProto& device_event_proto) {
  const MemcpyEventInfoProto& memcpy_info_proto =
      device_event_proto.memcpy_info();
  MemcpyEventInfo memcpy_info;
  memcpy_info.num_bytes = memcpy_info_proto.num_bytes();
  std::strncpy(memcpy_info.copy_kind,
               memcpy_info_proto.copy_kind().c_str(),
               phi::kMemKindMaxLen - 1);
  std::strncpy(memcpy_info.src_kind,
               memcpy_info_proto.src_kind().c_str(),
               phi::kMemKindMaxLen - 1);
  std::strncpy(memcpy_info.dst_kind,
               memcpy_info_proto.dst_kind().c_str(),
               phi::kMemKindMaxLen - 1);
  return memcpy_info;
}

MemsetEventInfo DeserializationReader::HandleMemsetEventInfoProto(
    const DeviceTraceEventProto& device_event_proto) {
  const MemsetEventInfoProto& memset_info_proto =
      device_event_proto.memset_info();
  MemsetEventInfo memset_info;
  memset_info.num_bytes = memset_info_proto.num_bytes();
  std::strncpy(memset_info.memory_kind,
               memset_info_proto.memory_kind().c_str(),
               phi::kMemKindMaxLen - 1);
  memset_info.value = memset_info_proto.value();
  return memset_info;
}

}  // namespace paddle::platform
