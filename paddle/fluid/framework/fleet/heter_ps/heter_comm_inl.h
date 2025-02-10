/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#ifdef PADDLE_WITH_HETERPS
#include <algorithm>
#include <memory>
#include <queue>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_kernel.h"
#include "paddle/phi/core/platform/device_context.h"
#ifdef PADDLE_WITH_XPU_KP
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#endif
#include "paddle/common/flags.h"

COMMON_DECLARE_double(gpugraph_hbm_table_load_factor);
COMMON_DECLARE_bool(gpugraph_enable_gpu_direct_access);
COMMON_DECLARE_bool(gpugraph_enable_segment_merge_grads);
COMMON_DECLARE_uint64(gpugraph_merge_grads_segment_size);
COMMON_DECLARE_int32(gpugraph_dedup_pull_push_mode);
COMMON_DECLARE_bool(enable_tracker_all2all);
COMMON_DECLARE_bool(enable_all2all_use_fp16);
COMMON_DECLARE_bool(enable_sparse_inner_gather);
COMMON_DECLARE_bool(graph_embedding_split_infer_mode);

namespace paddle {
namespace framework {
inline int64_t tick_usec() {
  struct timeval tm;
  gettimeofday(&tm, NULL);
  return tm.tv_sec * 1000 * 1000L + tm.tv_usec;
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
HeterComm<KeyType, ValType, GradType, GPUAccessor>::HeterComm(
    size_t capacity, std::shared_ptr<HeterPsResource> resource) {
  VLOG(1) << "Construct new HeterComm";
  resource_ = resource;
  device_num_ = resource_->total_device();
  storage_.resize(device_num_);
  multi_mf_dim_ = resource->multi_mf();
  load_factor_ = FLAGS_gpugraph_hbm_table_load_factor;
  multi_node_ = resource_->multi_node();
#if defined(PADDLE_WITH_CUDA)
  rdma_checker_ = GpuRDMAChecker::get(device_num_);
  topo_aware_ = rdma_checker_->topo_aware();
#endif
  enable_gpu_direct_access_ =
      (topo_aware_) ? false : FLAGS_gpugraph_enable_gpu_direct_access;
  VLOG(0) << "device_num = " << device_num_ << ", multi_node = " << multi_node_
          << ", multi_mf_dim = " << multi_mf_dim_
          << ", topo_aware = " << topo_aware_
          << ", enable_gpu_direct_access = " << enable_gpu_direct_access_
          << ", load_factor = " << load_factor_;
  if (multi_mf_dim_) {
    max_mf_dim_ = resource_->max_mf_dim();
    auto accessor_wrapper_ptr =
        GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
    val_type_size_ = accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);
    grad_type_size_ = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
    pull_type_size_ = accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);
    VLOG(0) << " HeterComm init, max_mf_dim: " << max_mf_dim_
            << ", max feature_value_size:" << val_type_size_
            << ", feature_value_push_size:" << grad_type_size_
            << ", feature_pull_type_size:" << pull_type_size_;
  } else {
    val_type_size_ = sizeof(ValType);
    pull_type_size_ = sizeof(ValType);
    grad_type_size_ = sizeof(GradType);
  }
  max_type_size_ = std::max(pull_type_size_, grad_type_size_);

  for (int i = 0; i < device_num_; ++i) {
    auto stream = resource_->local_stream(i, 0);
#if defined(PADDLE_WITH_CUDA)
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
        8, 1, (unsigned int)-1, (size_t)-1, false, false));  // NOLINT
#endif
    if (!multi_mf_dim_) {
      if (capacity > 0) {
#if defined(PADDLE_WITH_CUDA)
        auto table = new Table(capacity / load_factor_, stream);
#else
        auto table = new Table(capacity / load_factor_);
#endif
        tables_.push_back(table);
      }
    } else {
#if defined(PADDLE_WITH_CUDA)
      auto ptr_table = new PtrTable(capacity / load_factor_, stream);
#else
      auto ptr_table = new PtrTable(capacity / load_factor_);
#endif
      ptr_table->set_feature_value_size(pull_type_size_, grad_type_size_);
      ptr_tables_.push_back(ptr_table);
    }
    if (multi_node_) {
      storage_[i].init(device_num_,
                       resource_->dev_id(i),
                       phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    }
  }
  barrier_.reset(device_num_);
  heter_comm_kernel_ = std::make_unique<HeterCommKernel>(block_size_);
  init_path();
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
HeterComm<KeyType, ValType, GradType, GPUAccessor>::HeterComm(
    size_t capacity,
    std::shared_ptr<HeterPsResource> resource,
    GPUAccessor &gpu_accessor) {  // NOLINT
  VLOG(1) << "Construct new HeterComm";
  resource_ = resource;
  device_num_ = resource_->total_device();
  storage_.resize(device_num_);
  multi_mf_dim_ = resource->multi_mf();
  gpu_accessor_ = gpu_accessor;
  load_factor_ = FLAGS_gpugraph_hbm_table_load_factor;
  multi_node_ = resource_->multi_node();
#if defined(PADDLE_WITH_CUDA)
  rdma_checker_ = GpuRDMAChecker::get(device_num_);
  topo_aware_ = rdma_checker_->topo_aware();
#endif
  enable_gpu_direct_access_ =
      (topo_aware_) ? false : FLAGS_gpugraph_enable_gpu_direct_access;
  VLOG(0) << "gpu access device_num = " << device_num_
          << ", multi_node = " << multi_node_
          << ", multi_mf_dim = " << multi_mf_dim_
          << ", topo_aware = " << topo_aware_
          << ", enable_gpu_direct_access = " << enable_gpu_direct_access_
          << ", load_factor = " << load_factor_
          << ", graph_embedding_split_infer_mode="
          << FLAGS_graph_embedding_split_infer_mode;
  if (multi_mf_dim_) {
    max_mf_dim_ = resource_->max_mf_dim();
    auto accessor_wrapper_ptr =
        GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
    val_type_size_ = accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);
    grad_type_size_ = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
    pull_type_size_ = accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);
    VLOG(0) << " HeterComm init, max_mf_dim: " << max_mf_dim_
            << ", max feature_value_size:" << val_type_size_
            << ", feature_value_push_size:" << grad_type_size_
            << ", feature_pull_type_size:" << pull_type_size_;
  } else {
    val_type_size_ = sizeof(ValType);
    pull_type_size_ = sizeof(ValType);
    grad_type_size_ = sizeof(GradType);
  }
  max_type_size_ = std::max(pull_type_size_, grad_type_size_);

  for (int i = 0; i < device_num_; ++i) {
    auto stream = resource_->local_stream(i, 0);
#if defined(PADDLE_WITH_CUDA)
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
        8, 1, (unsigned int)-1, (size_t)-1, false, false));  // NOLINT
#endif
    if (!multi_mf_dim_) {
#if defined(PADDLE_WITH_CUDA)
      auto table = new Table(capacity / load_factor_, stream);
#else
      auto table = new Table(capacity / load_factor_);
#endif
      tables_.push_back(table);
    } else {
#if defined(PADDLE_WITH_CUDA)
      auto ptr_table = new PtrTable(capacity / load_factor_, stream);
#else
      auto ptr_table = new PtrTable(capacity / load_factor_);
#endif
      ptr_table->set_feature_value_size(pull_type_size_, grad_type_size_);
      ptr_tables_.push_back(ptr_table);
    }
    if (multi_node_) {
      storage_[i].init(device_num_,
                       resource_->dev_id(i),
                       phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    }
  }
  barrier_.reset(device_num_);
  heter_comm_kernel_ = std::make_unique<HeterCommKernel>(block_size_);
  init_path();
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::init_path() {
  int total_device = resource_->total_device();
  path_.resize(total_device);
  if (!topo_aware_) {
    VLOG(0) << "init path without topo aware";
    for (int i = 0; i < total_device; ++i) {
      path_[i].resize(total_device);
      for (int j = 0; j < total_device; ++j) {
        auto &nodes = path_[i][j].nodes_;
        nodes.resize(1);
        nodes[0].in_stream = resource_->remote_stream(i, j);   // i->j
        nodes[0].out_stream = resource_->remote_stream(j, i);  // j->i
        nodes[0].key_storage = NULL;
        nodes[0].val_storage = NULL;
        nodes[0].sync = 0;
        nodes[0].dev_num = j;
      }
    }
  } else {
    VLOG(0) << "init path with topo aware";
    for (int i = 0; i < total_device; ++i) {
      path_[i].resize(total_device);
      for (int j = 0; j < total_device; ++j) {
        auto &nodes = path_[i][j].nodes_;
        int from = resource_->dev_id(i);
        int to = resource_->dev_id(j);

        int transfer_id = i;
        if (need_transfer(from, to)) {
          transfer_id = resource_->get_index_by_devid(get_transfer_devid(from));
          nodes.push_back(Node());
          Node &node = nodes.back();
          node.in_stream = resource_->remote_stream(i, transfer_id);
          node.out_stream = resource_->remote_stream(transfer_id, i);
          node.key_storage = NULL;
          node.val_storage = NULL;
          node.sync = 1;
          node.dev_num = transfer_id;
        }
        nodes.push_back(Node());
        Node &node = nodes.back();
        node.in_stream = resource_->remote_stream(transfer_id, j);
        node.out_stream = resource_->remote_stream(j, transfer_id);
        node.key_storage = NULL;
        node.val_storage = NULL;
        node.sync = 1;
        node.dev_num = j;
      }
    }
  }
  start_time_ = tick_usec();
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::reset_table(
    const int dev_id,
    size_t capacity,
    const OptimizerConfig &sgd_config,
    const OptimizerConfig &embedx_config,
    bool infer_mode) {
  PADDLE_ENFORCE_LT(
      dev_id,
      device_num_,
      common::errors::InvalidArgument(
          "dev id %d more than device num %d", dev_id, device_num_));
#if defined(PADDLE_WITH_CUDA)
  platform::CUDADeviceGuard guard(resource_->dev_id(dev_id));
  auto stream = resource_->local_stream(dev_id, 0);
#endif
  size_t need_capacity = capacity / load_factor_;
  if (!multi_mf_dim_) {
    auto table = tables_[dev_id];
    if (static_cast<size_t>(table->size()) < need_capacity) {
      delete table;
#if defined(PADDLE_WITH_CUDA)
      table = new Table(need_capacity, stream);
#else
      table = new Table(need_capacity);
#endif
      table->set_sparse_sgd(sgd_config);
      table->set_embedx_sgd(sgd_config);
      tables_[dev_id] = table;
    } else {
      table->clear(stream);
    }
    table->set_mode(infer_mode);
  } else {
    auto table = ptr_tables_[dev_id];
    if (static_cast<size_t>(table->size()) < need_capacity) {
      delete table;
#if defined(PADDLE_WITH_CUDA)
      table = new PtrTable(need_capacity, stream);
#else
      table = new PtrTable(need_capacity);
#endif
      table->set_feature_value_size(pull_type_size_, grad_type_size_);
      table->set_sparse_sgd(sgd_config);
      table->set_embedx_sgd(sgd_config);
      ptr_tables_[dev_id] = table;
    } else {
      table->clear(stream);
    }
    table->set_mode(infer_mode);
  }
  is_infer_mode_ = infer_mode;
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::set_mode(
    bool infer_mode) {
  if (!multi_mf_dim_) {
    for (auto &table : tables_) {
      table->set_mode(infer_mode);
    }
  } else {
    for (auto &table : ptr_tables_) {
      table->set_mode(infer_mode);
    }
  }
  is_infer_mode_ = infer_mode;
}
// debug time
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::print_debug_time(
    const int &gpu_id, bool force) {
  if (!multi_node_) {
    return;
  }
  static int64_t count_ = 0;
  if ((count_++ % 5000) != 0) {
    return;
  }
  auto &cc = storage_[gpu_id];
  printf(
      "gpu id=%d, count=%ld, "
      "keys: %lu %lu %lu, "
      "all2all: %lf, node span: %lf, wait: %lf trans:%lf p2p:%lf barrier: %lf, "
      "inner span: %lf, barrier: %lf, "
      "local op: %lf\n",
      gpu_id,
      count_++,
      cc.total_keys_,
      cc.local_keys_,
      cc.remote_keys_,
      cc.all2all_span_.ElapsedSec(),
      cc.node_span_.ElapsedSec(),
      cc.node_wait_.ElapsedSec(),
      cc.node_trans_.ElapsedSec(),
      cc.node_p2p_.ElapsedSec(),
      cc.node_barrier_.ElapsedSec(),
      cc.inner_span_.ElapsedSec(),
      cc.inner_barrier_.ElapsedSec(),
      cc.local_oper_.ElapsedSec());
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename DstPlace, typename SrcPlace, typename StreamType>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::memory_copy(
    DstPlace dst_place,
    void *dst,
    SrcPlace src_place,
    const void *src,
    size_t count,
    StreamType stream) {
#if defined(PADDLE_WITH_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream));
  if (stream == 0) {
    CUDA_CHECK(cudaStreamSynchronize(0));
  }
#elif defined(PADDLE_WITH_XPU_KP)
  memory::Copy(dst_place, dst, src_place, src, count);
#endif
}

#if defined(PADDLE_WITH_CUDA)
inline int get_dev_by_ptr(const void *ptr) {
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  int dev = -1;
#if CUDART_VERSION >= 10000
  if (attr.type == cudaMemoryTypeDevice)
#else
  if (attr.memoryType == cudaMemoryTypeDevice)
#endif
  {
    dev = attr.device;
  }
  return dev;
}
#endif
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename StreamType>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::MemcpyPeerAsync(
    void *dst, const void *src, size_t count, StreamType stream) {
#if defined(PADDLE_WITH_CUDA)
  int src_device = get_dev_by_ptr(src);
  int dst_device = get_dev_by_ptr(dst);
  AnyDeviceGuard guard(resource_->dev_id(src_device));
  if (dst_device == -1 || src_device == -1) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream));
  } else if (dst_device == src_device) {
    CUDA_CHECK(
        cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream));
  } else {
    CUDA_CHECK(
        cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream));
  }
#endif
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::create_storage(
    int start_index, int end_index, size_t keylen, size_t vallen) {
#if defined(PADDLE_WITH_CUDA)
  auto &allocator = allocators_[start_index];
  auto &nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].dev_num));
    if (keylen > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceAllocate(
          resource_->dev_id(nodes[i].dev_num),
          (void **)&(nodes[i].key_storage),  // NOLINT
          keylen,
          resource_->remote_stream(nodes[i].dev_num, start_index)));
      nodes[i].key_bytes_len = keylen;
    }
    if (vallen > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceAllocate(
          resource_->dev_id(nodes[i].dev_num),
          (void **)&(nodes[i].val_storage),  // NOLINT
          vallen,
          resource_->remote_stream(nodes[i].dev_num, start_index)));
      nodes[i].val_bytes_len = vallen;
    }
  }
#elif defined(PADDLE_WITH_XPU_KP)
  auto &nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    phi::backends::xpu::XPUDeviceGuard guard(
        resource_->dev_id(nodes[i].dev_num));
    auto place = DevPlace(resource_->dev_id(nodes[i].dev_num));
    if (keylen > 0) {
      auto node_keys_mem = MemoryAlloc(place, keylen);
      nodes[i].key_storage = reinterpret_cast<char *>(node_keys_mem->ptr());
      nodes[i].key_bytes_len = keylen;
    }
    if (vallen > 0) {
      auto node_vals_mem = MemoryAlloc(place, vallen);
      nodes[i].val_storage = reinterpret_cast<char *>(node_vals_mem->ptr());
      nodes[i].val_bytes_len = vallen;
    }
  }
#endif
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::create_tmp_storage(
    void *&dest, int start_index, int end_index, size_t vallen) {  // NOLINT
#if defined(PADDLE_WITH_CUDA)
  auto &allocator = allocators_[start_index];
  platform::CUDADeviceGuard guard(resource_->dev_id(end_index));
  PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceAllocate(
      resource_->dev_id(end_index),
      reinterpret_cast<void **>(&dest),
      vallen,
      resource_->remote_stream(end_index, start_index)));

#elif defined(PADDLE_WITH_XPU_KP)
  phi::backends::xpu::XPUDeviceGuard guard(resource_->dev_id(end_index));
  auto place = DevPlace(resource_->dev_id(end_index));
  auto node_vals_mem = MemoryAlloc(place, vallen);
  dest = reinterpret_cast<void *>(node_vals_mem->ptr());
#endif
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::destroy_storage(
    int start_index, int end_index) {
#if defined(PADDLE_WITH_CUDA)
  auto &allocator = allocators_[start_index];
  auto &nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].dev_num));

    PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceFree(
        resource_->dev_id(nodes[i].dev_num), nodes[i].key_storage));
    PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceFree(
        resource_->dev_id(nodes[i].dev_num), nodes[i].val_storage));
  }
#endif
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::destroy_tmp_storage(
    void *&p, int start_index, int end_index) {  // NOLINT
#if defined(PADDLE_WITH_CUDA)
  auto &allocator = allocators_[start_index];
  platform::CUDADeviceGuard guard(resource_->dev_id(end_index));
  PADDLE_ENFORCE_GPU_SUCCESS(
      allocator->DeviceFree(resource_->dev_id(end_index), p));
#endif
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::walk_to_dest(
    int start_index,
    int num,
    int *h_left,
    int *h_right,
    KeyType *src_key,
    GradType *src_val) {
  AnyDeviceGuard guard(resource_->dev_id(start_index));
  for (int i = 0; i < num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto &nodes = path_[start_index][i].nodes_;
    auto &node = nodes[0];
    MemcpyPeerAsync(node.key_storage,
                    reinterpret_cast<char *>(src_key + h_left[i]),
                    node.key_bytes_len,
                    node.in_stream);
    if (src_val) {
      MemcpyPeerAsync(node.val_storage,
                      reinterpret_cast<char *>(src_val + h_left[i]),
                      node.val_bytes_len,
                      node.in_stream);
    }
    // transfer
    int step_num = static_cast<int>(nodes.size()) - 1;
    if (step_num == 0) {
      continue;
    }
    if (node.sync) {
      sync_stream(node.in_stream);
    }
    for (int cur_step = 0; cur_step < step_num; ++cur_step) {
      auto &src_node = nodes[cur_step];
      auto &dst_node = nodes[cur_step + 1];
      MemcpyPeerAsync(dst_node.key_storage,
                      src_node.key_storage,
                      dst_node.key_bytes_len,
                      src_node.in_stream);
      if (src_val) {
        MemcpyPeerAsync(dst_node.val_storage,
                        src_node.val_storage,
                        dst_node.val_bytes_len,
                        src_node.in_stream);
      }
      if (src_node.sync) {
        sync_stream(src_node.in_stream);
      }
    }
  }
  // wait stream to finish
  for (int i = 0; i < num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto &node = path_[start_index][i].nodes_.back();
    sync_stream(node.in_stream);
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::walk_to_dest(
    int start_index,
    int gpu_num,
    int *h_left,
    int *h_right,
    KeyType *src_key,
    char *src_val,
    size_t val_size) {
  AnyDeviceGuard guard(resource_->dev_id(start_index));
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto &nodes = path_[start_index][i].nodes_;
    auto &node = nodes[0];
    MemcpyPeerAsync(node.key_storage,
                    reinterpret_cast<char *>(src_key + h_left[i]),
                    node.key_bytes_len,
                    node.in_stream);
    if (src_val) {
      MemcpyPeerAsync(node.val_storage,
                      src_val + uint64_t(h_left[i]) * uint64_t(val_size),
                      node.val_bytes_len,
                      node.in_stream);
    }
    int step_num = static_cast<int>(nodes.size()) - 1;
    if (step_num == 0) {
      continue;
    }
    if (node.sync) {
      sync_stream(node.in_stream);
    }
    // transfer
    for (int cur_step = 0; cur_step < step_num; ++cur_step) {
      auto &src_node = nodes[cur_step];
      auto &dest_node = nodes[cur_step + 1];
      MemcpyPeerAsync(dest_node.key_storage,
                      src_node.key_storage,
                      src_node.key_bytes_len,
                      src_node.in_stream);
      if (src_val) {
        MemcpyPeerAsync(dest_node.val_storage,
                        src_node.val_storage,
                        src_node.val_bytes_len,
                        src_node.in_stream);
      }
      if (src_node.sync) {
        sync_stream(src_node.in_stream);
      }
    }
  }
  // wait stream to finish
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto &node = path_[start_index][i].nodes_.back();
    sync_stream(node.in_stream);
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::walk_to_src(
    int start_index,
    int gpu_num,
    int *h_left,
    int *h_right,
    char *src_val,
    size_t val_size) {
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    AnyDeviceGuard guard(resource_->dev_id(i));
    auto &nodes = path_[start_index][i].nodes_;
    int step_num = static_cast<int>(nodes.size() - 1);
    if (step_num > 0) {
      // transfer
      for (int cur_step = step_num; cur_step > 0; --cur_step) {
        auto &src_node = nodes[cur_step];
        auto &dst_node = nodes[cur_step - 1];
        MemcpyPeerAsync(dst_node.val_storage,
                        src_node.val_storage,
                        dst_node.val_bytes_len,
                        src_node.out_stream);
        if (src_node.sync) {
          sync_stream(src_node.out_stream);
        }
      }
    }
    auto &node = nodes[0];
    MemcpyPeerAsync(src_val + uint64_t(h_left[i]) * val_size,
                    node.val_storage,
                    node.val_bytes_len,
                    node.out_stream);
  }
  // wait stream to finish
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    AnyDeviceGuard guard(resource_->dev_id(i));
    auto &node = path_[start_index][i].nodes_.front();
    sync_stream(node.out_stream);
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
HeterComm<KeyType, ValType, GradType, GPUAccessor>::~HeterComm() {
  for (int i = 0; i < device_num_; ++i) {
    print_debug_time(i, true);
  }
  if (!multi_mf_dim_) {
    for (auto &table : tables_) {
      delete table;
      table = nullptr;
    }
  } else {
    for (auto &table : ptr_tables_) {
      delete table;
      table = nullptr;
    }
    for (auto &table : tables_) {
      delete table;
      table = nullptr;
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::show_one_table(
    int gpu_num) {
  if (!multi_mf_dim_) {
    tables_[gpu_num]->show();
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    show_table_collisions() {
  size_t idx = 0;
  for (auto &table : tables_) {
    if (table != nullptr) {
      table->show_collision(idx++);
    }
  }
  idx = 0;
  for (auto &table : ptr_tables_) {
    if (table != nullptr) {
      table->show_collision(idx++);
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::log2i(int x) {
  unsigned res = 0;
  while (x >>= 1) {
    ++res;
  }
  return res;
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::get_index_by_devid(
    int devid) {
  return resource_->get_index_by_devid(devid);
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::set_sparse_sgd(
    const OptimizerConfig &optimizer_config) {
  for (int i = 0; i < resource_->total_device(); ++i) {
    AnyDeviceGuard guard(resource_->dev_id(i));
    if (!multi_mf_dim_) {
      tables_[i]->set_sparse_sgd(optimizer_config);
    } else {
      ptr_tables_[i]->set_sparse_sgd(optimizer_config);
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::set_embedx_sgd(
    const OptimizerConfig &optimizer_config) {
  for (int i = 0; i < resource_->total_device(); ++i) {
    AnyDeviceGuard guard(resource_->dev_id(i));
    if (!multi_mf_dim_) {
      tables_[i]->set_embedx_sgd(optimizer_config);
    } else {
      ptr_tables_[i]->set_embedx_sgd(optimizer_config);
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::build_ps(
    int dev_num,
    KeyType *h_keys,
    ValType *h_vals,
    size_t len,
    size_t chunk_size,
    int stream_num,
    int offset) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(dev_num);

  std::vector<std::shared_ptr<phi::Allocation>> d_key_bufs;
  std::vector<std::shared_ptr<phi::Allocation>> d_val_bufs;

  // auto adjust stream num by data length
  int max_stream = (len + chunk_size - 1) / chunk_size;
  if (max_stream < stream_num) {
    stream_num = max_stream;
  }
  if (stream_num > device_num_) {
    stream_num = device_num_;
  }

  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  ppStream streams[stream_num];  // NOLINT

  d_key_bufs.resize(stream_num);
  d_val_bufs.resize(stream_num);
  for (int i = 0; i < stream_num; ++i) {
    streams[i] = resource_->local_stream(dev_num, i);
    d_key_bufs[i] = MemoryAlloc(place, chunk_size * sizeof(KeyType));
    d_val_bufs[i] = MemoryAlloc(place, chunk_size * sizeof(ValType));
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (static_cast<size_t>(cur_len) < len) {
    cur_stream = cur_stream % stream_num;
    auto cur_use_stream = streams[cur_stream];
#if defined(PADDLE_WITH_XPU_KP)
    cur_use_stream = 0;
#endif

    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;

    auto dst_place = place;
    auto src_place = phi::CPUPlace();

    memory_copy(dst_place,
                reinterpret_cast<char *>(d_key_bufs[cur_stream]->ptr()),
                src_place,
                h_keys + cur_len,
                sizeof(KeyType) * tmp_len,
                cur_use_stream);
    memory_copy(dst_place,
                reinterpret_cast<char *>(d_val_bufs[cur_stream]->ptr()),
                src_place,
                h_vals + cur_len,
                sizeof(ValType) * tmp_len,
                cur_use_stream);
    if (offset == -1) offset = dev_num;
    tables_[offset]->insert(
        reinterpret_cast<KeyType *>(d_key_bufs[cur_stream]->ptr()),
        reinterpret_cast<ValType *>(d_val_bufs[cur_stream]->ptr()),
        static_cast<size_t>(tmp_len),
        cur_use_stream);

    cur_stream += 1;
    cur_len += tmp_len;
  }
  for (int i = 0; i < stream_num; ++i) {
    sync_stream(streams[i]);
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::build_ps(
    int num,
    KeyType *h_keys,
    char *pool,
    size_t len,
    size_t feature_value_size,
    size_t chunk_size,
    int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(num);

  // auto adjust stream num by data length
  int max_stream = (len + chunk_size - 1) / chunk_size;
  if (max_stream < stream_num) {
    stream_num = max_stream;
  }
  if (stream_num > device_num_) {
    stream_num = device_num_;
  }

  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);

  // use hbm pool
  std::vector<std::shared_ptr<phi::Allocation>> d_key_bufs;

  ppStream streams[stream_num];  // NOLINT
  d_key_bufs.resize(stream_num);
  for (int i = 0; i < stream_num; ++i) {
    streams[i] = resource_->local_stream(num, i);
    d_key_bufs[i] = MemoryAlloc(place, chunk_size * sizeof(KeyType));
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (cur_len < len) {
    cur_stream = cur_stream % stream_num;
    auto cur_use_stream = streams[cur_stream];
#if defined(PADDLE_WITH_XPU_KP)
    cur_use_stream = 0;
#endif
    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;

    auto dst_place = place;
    auto src_place = phi::CPUPlace();

    memory_copy(dst_place,
                reinterpret_cast<char *>(d_key_bufs[cur_stream]->ptr()),
                src_place,
                h_keys + cur_len,
                sizeof(KeyType) * tmp_len,
                cur_use_stream);
    ptr_tables_[num]->insert(
        reinterpret_cast<KeyType *>(d_key_bufs[cur_stream]->ptr()),
        tmp_len,
        pool,
        feature_value_size,
        cur_len,
        cur_use_stream);
    cur_stream += 1;
    cur_len += tmp_len;
  }
  for (int i = 0; i < stream_num; ++i) {
    sync_stream(streams[i]);
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::merge_grad(
    int dev_num,
    KeyType *d_keys,
    GradType *d_grads,
    size_t len,
    int &uniq_len) {  // NOLINT
  int dev_id = resource_->dev_id(dev_num);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(dev_num, 0);
  size_t temp_storage_bytes;
  auto d_merge_keys = MemoryAlloc(place, len * sizeof(KeyType));
  KeyType *d_merge_keys_ptr = reinterpret_cast<KeyType *>(d_merge_keys->ptr());
  auto d_merge_grads = MemoryAlloc(place, len * sizeof(GradType));
  GradType *d_merge_grads_ptr =
      reinterpret_cast<GradType *>(d_merge_grads->ptr());
  heter_comm_kernel_->sort_pairs(NULL,
                                 temp_storage_bytes,
                                 d_keys,
                                 d_merge_keys_ptr,
                                 d_grads,
                                 d_merge_grads_ptr,
                                 len,
                                 dev_id,
                                 0,
                                 8 * sizeof(KeyType),
                                 stream,
                                 false);
  auto d_temp_storage = MemoryAlloc(place, temp_storage_bytes);
  heter_comm_kernel_->sort_pairs(d_temp_storage->ptr(),
                                 temp_storage_bytes,
                                 d_keys,
                                 d_merge_keys_ptr,
                                 d_grads,
                                 d_merge_grads_ptr,
                                 len,
                                 dev_id,
                                 0,
                                 8 * sizeof(KeyType),
                                 stream,
                                 false);
  temp_storage_bytes = 0;
  auto d_num_runs_out_mem = MemoryAlloc(place, sizeof(int));
  int *d_num_runs_out = reinterpret_cast<int *>(d_num_runs_out_mem->ptr());
  heter_comm_kernel_->reduce_by_key(NULL,
                                    temp_storage_bytes,
                                    d_merge_keys_ptr,
                                    d_keys,
                                    d_merge_grads_ptr,
                                    d_grads,
                                    d_num_runs_out,
                                    len,
                                    stream,
                                    false);
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = MemoryAlloc(place, temp_storage_bytes);
  }
  heter_comm_kernel_->reduce_by_key(d_temp_storage->ptr(),
                                    temp_storage_bytes,
                                    d_merge_keys_ptr,
                                    d_keys,
                                    d_merge_grads_ptr,
                                    d_grads,
                                    d_num_runs_out,
                                    len,
                                    stream,
                                    false);
  auto dst_place = phi::CPUPlace();
  auto src_place = place;
  memory_copy(
      dst_place, &uniq_len, src_place, d_num_runs_out, sizeof(int), stream);
  sync_stream(stream);
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::dynamic_merge_grad(
    int gpu_num,
    KeyType *d_keys,
    float *d_grads,
    size_t len,
    int &uniq_len,        // NOLINT
    size_t &segment_len,  // NOLINT
    bool enable_segment_merge_grad) {
  int dev_id = resource_->dev_id(gpu_num);
  phi::GPUPlace place = phi::GPUPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t temp_storage_bytes;
  size_t grad_dim = max_mf_dim_;
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  auto d_merge_keys = MemoryAlloc(place, len * sizeof(KeyType));
  KeyType *d_merge_keys_ptr = reinterpret_cast<KeyType *>(d_merge_keys->ptr());
  auto d_fea_num_info = MemoryAlloc(place, sizeof(uint32_t) * (len * 3 + 1));
  uint32_t *d_fea_num_info_ptr =
      reinterpret_cast<uint32_t *>(d_fea_num_info->ptr());
  uint32_t *d_index = static_cast<uint32_t *>(&d_fea_num_info_ptr[len]);
  uint32_t *d_idx = reinterpret_cast<uint32_t *>(&d_index[len]);
  int *d_merged_size = reinterpret_cast<int *>(&d_idx[len]);
  heter_comm_kernel_->fill_idx(d_idx, len, stream, dev_id);

  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(NULL,
                                      temp_storage_bytes,
                                      d_keys,
                                      d_merge_keys_ptr,
                                      d_idx,
                                      d_index,
                                      len,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream));
  auto d_temp_storage = MemoryAlloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(d_temp_storage->ptr(),
                                      temp_storage_bytes,
                                      d_keys,
                                      d_merge_keys_ptr,
                                      d_idx,
                                      d_index,
                                      len,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(NULL,
                                         temp_storage_bytes,
                                         d_merge_keys_ptr,
                                         d_keys,
                                         d_fea_num_info_ptr,
                                         d_merged_size,
                                         len,
                                         stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = MemoryAlloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(d_temp_storage->ptr(),
                                         temp_storage_bytes,
                                         d_merge_keys_ptr,
                                         d_keys,
                                         d_fea_num_info_ptr,
                                         d_merged_size,
                                         len,
                                         stream));

  cudaMemcpyAsync(reinterpret_cast<void *>(&uniq_len),
                  d_merged_size,
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  assert(d_merged_size > 0);
  uint32_t *d_offset = reinterpret_cast<uint32_t *>(&d_index[len]);
  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(NULL,
                                                           temp_storage_bytes,
                                                           d_fea_num_info_ptr,
                                                           d_offset,
                                                           uniq_len,
                                                           stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = MemoryAlloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceScan::ExclusiveSum(d_temp_storage->ptr(),
                                    temp_storage_bytes,
                                    d_fea_num_info_ptr,
                                    d_offset,
                                    uniq_len,
                                    stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  if (enable_segment_merge_grad) {
    segment_merge_grad(gpu_num,
                       d_merge_keys_ptr,
                       d_grads,
                       d_index,
                       len,
                       d_fea_num_info_ptr,
                       uniq_len,
                       segment_len);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(d_keys,
                                               d_merge_keys_ptr,
                                               sizeof(KeyType) * segment_len,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  } else {
    auto d_merge_grads = MemoryAlloc(place, len * grad_value_size);
    float *d_merge_grads_ptr = reinterpret_cast<float *>(d_merge_grads->ptr());
    // copy merge keys to d_keys
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(d_keys,
                                               d_merge_keys_ptr,
                                               sizeof(KeyType) * segment_len,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
    heter_comm_kernel_->merge_gradient(
        d_keys,
        d_offset,
        d_fea_num_info_ptr,
        d_index,
        reinterpret_cast<char *>(d_grads),
        reinterpret_cast<char *>(d_merge_grads_ptr),
        uniq_len,
        grad_dim,
        grad_value_size,
        merger_,
        stream,
        gpu_accessor_);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(d_grads,
                                               d_merge_grads_ptr,
                                               grad_value_size * uniq_len,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::segment_merge_grad(
    int gpu_num,  // the device number
    KeyType
        *d_keys,  // the sorted keys list, which will be modified after merged
    float *d_grads,  // the raw grads list, which will be modified after merged
    const uint32_t
        *d_index,  // the storage position of d_keys, its length is len.
    size_t len,    // the number of raw input keys
    const uint32_t
        *d_fea_num_info,     // prefix sum array, its length is uniq_len+1
    size_t uniq_len,         // the number of unique keys
    size_t &segments_num) {  // the number of segment merged keys // NOLINT

  int dev_id = resource_->dev_id(gpu_num);
  phi::GPUPlace place = phi::GPUPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  auto grad_dim = max_mf_dim_;
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  auto d_buffer1 = MemoryAlloc(place, sizeof(uint32_t) * len);
  auto d_segments = reinterpret_cast<uint32_t *>(d_buffer1->ptr());
  auto d_buffer2 = MemoryAlloc(place, sizeof(uint32_t) * len);
  auto d_segments_offset = reinterpret_cast<uint32_t *>(d_buffer2->ptr());
  auto d_buffer3 = MemoryAlloc(place, sizeof(uint32_t) * len);
  auto d_segments_fea_num_info = reinterpret_cast<uint32_t *>(d_buffer3->ptr());
  auto d_buffer4 = MemoryAlloc(place, sizeof(uint32_t) * len);
  auto d_segments_fea_num_offset =
      reinterpret_cast<uint32_t *>(d_buffer4->ptr());
  auto d_buffer5 = MemoryAlloc(place, sizeof(uint32_t));
  auto d_segments_num = reinterpret_cast<uint32_t *>(d_buffer5->ptr());
  CUDA_CHECK(cudaMemsetAsync(d_segments_num, 0, sizeof(uint32_t), stream));

  uint32_t segment_size = FLAGS_gpugraph_merge_grads_segment_size;
  heter_comm_kernel_->split_segments(d_fea_num_info,
                                     uniq_len,
                                     d_segments,
                                     d_segments_num,
                                     segment_size,
                                     stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  size_t temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::Sum(
      NULL, temp_storage_bytes, d_segments, d_segments_num, uniq_len, stream));
  auto d_temp_storage = MemoryAlloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::Sum(d_temp_storage->ptr(),
                                                    temp_storage_bytes,
                                                    d_segments,
                                                    d_segments_num,
                                                    uniq_len,
                                                    stream));
  CUDA_CHECK(cudaMemcpyAsync(&segments_num,
                             d_segments_num,
                             sizeof(uint32_t),
                             cudaMemcpyDeviceToHost,
                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(NULL,
                                                           temp_storage_bytes,
                                                           d_segments,
                                                           d_segments_offset,
                                                           uniq_len,
                                                           stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = MemoryAlloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceScan::ExclusiveSum(d_temp_storage->ptr(),
                                    temp_storage_bytes,
                                    d_segments,
                                    d_segments_offset,
                                    uniq_len,
                                    stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  heter_comm_kernel_->expand_segments(d_fea_num_info,
                                      d_segments_offset,
                                      uniq_len,
                                      d_segments_fea_num_info,
                                      segment_size,
                                      stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceScan::ExclusiveSum(NULL,
                                    temp_storage_bytes,
                                    d_segments_fea_num_info,
                                    d_segments_fea_num_offset,
                                    segments_num,
                                    stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = MemoryAlloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceScan::ExclusiveSum(d_temp_storage->ptr(),
                                    temp_storage_bytes,
                                    d_segments_fea_num_info,
                                    d_segments_fea_num_offset,
                                    segments_num,
                                    stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  auto d_segments_keys = MemoryAlloc(place, sizeof(KeyType) * segments_num);
  auto d_segments_keys_ptr =
      reinterpret_cast<KeyType *>(d_segments_keys->ptr());
  heter_comm_kernel_->shrink_keys(d_keys,
                                  d_segments_fea_num_offset,
                                  d_segments_keys_ptr,
                                  segments_num,
                                  stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  auto d_segment_grads = MemoryAlloc(place, segments_num * grad_value_size);
  auto d_segment_grads_ptr = reinterpret_cast<float *>(d_segment_grads->ptr());
  heter_comm_kernel_->merge_gradient(
      d_segments_keys_ptr,
      d_segments_fea_num_offset,
      d_segments_fea_num_info,
      d_index,
      reinterpret_cast<char *>(d_grads),
      reinterpret_cast<char *>(d_segment_grads_ptr),
      segments_num,
      grad_dim,
      grad_value_size,
      merger_,
      stream,
      gpu_accessor_);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(d_keys,
                                             d_segments_keys_ptr,
                                             sizeof(KeyType) * segments_num,
                                             cudaMemcpyDeviceToDevice,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(d_grads,
                                             d_segment_grads_ptr,
                                             grad_value_size * segments_num,
                                             cudaMemcpyDeviceToDevice,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::split_input_to_shard(
    KeyType *d_keys,
    int *d_idx_ptr,
    size_t len,
    int *left,
    int *right,
    int dev_num) {
  auto stream = resource_->local_stream(dev_num, 0);
  split_idx_to_shard(d_keys, d_idx_ptr, len, left, right, dev_num, stream);
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename T, typename StreamType>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::split_idx_to_shard(
    KeyType *d_keys,
    T *d_idx_ptr,
    size_t len,
    T *left,
    T *right,
    int gpu_num,
    StreamType stream) {
  int total_device = resource_->total_device();
  int dev_id = resource_->dev_id(gpu_num);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);

  thread_local std::shared_ptr<memory::Allocation> d_idx_tmp = nullptr;
  T *d_idx_tmp_ptr = AllocCache<T>(&d_idx_tmp, place, 3 * len * sizeof(T));
  T *d_shard_index_ptr = reinterpret_cast<T *>(&d_idx_tmp_ptr[len]);
  T *d_shard_index_tmp_ptr = reinterpret_cast<T *>(&d_shard_index_ptr[len]);

  heter_comm_kernel_->fill_idx(d_idx_tmp_ptr, len, stream, dev_id);
  heter_comm_kernel_->calc_shard_index(
      d_keys, len, d_shard_index_tmp_ptr, total_device, stream, dev_id);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_device);
  heter_comm_kernel_->sort_pairs(NULL,
                                 temp_storage_bytes,
                                 d_shard_index_tmp_ptr,
                                 d_shard_index_ptr,
                                 d_idx_tmp_ptr,
                                 d_idx_ptr,
                                 len,
                                 dev_id,
                                 0,
                                 num_bits,
                                 stream);

  thread_local std::shared_ptr<memory::Allocation> d_temp_storage = nullptr;
  void *d_buf = AllocCache<void>(&d_temp_storage, place, temp_storage_bytes);
  heter_comm_kernel_->sort_pairs(d_buf,
                                 temp_storage_bytes,
                                 d_shard_index_tmp_ptr,
                                 d_shard_index_ptr,
                                 d_idx_tmp_ptr,
                                 d_idx_ptr,
                                 len,
                                 dev_id,
                                 0,
                                 num_bits,
                                 stream);

  heter_comm_kernel_->calc_shard_offset(
      d_shard_index_ptr, left, right, len, total_device, stream, dev_id);
  sync_stream(stream);
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename StreamType>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::merge_keys(
    const int gpu_num,
    const KeyType *d_keys,
    const size_t &len,       // input
    KeyType *d_sorted_keys,  // output
    KeyType *d_merged_keys,  // output
    uint32_t *d_restore_idx,
    StreamType stream) {
#if defined(PADDLE_WITH_CUDA)
  int dev_id = resource_->dev_id(gpu_num);
  phi::GPUPlace place = phi::GPUPlace(dev_id);

  thread_local std::shared_ptr<memory::Allocation> d_fea_num_info = nullptr;
  uint32_t *d_offset = AllocCache<uint32_t>(
      &d_fea_num_info, place, sizeof(uint32_t) * (len * 3));
  uint32_t *d_merged_cnts = &d_offset[len];
  uint32_t *d_sorted_idx = &d_merged_cnts[len];

  return dedup_keys_and_fillidx(gpu_num,
                                len,
                                d_keys,         // input
                                d_merged_keys,  // output
                                d_sorted_keys,
                                d_restore_idx,
                                d_sorted_idx,
                                d_offset,
                                d_merged_cnts,
                                false,
                                stream);
#else
  return 0;
#endif
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_merge_sparse(
    const int num, KeyType *d_keys, float *d_vals, size_t len) {
  int total_device = resource_->total_device();
  int dev_id = resource_->dev_id(num);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);

  int h_left[total_device];   // NOLINT
  int h_right[total_device];  // NOLINT

  auto d_left = MemoryAlloc(place, total_device * sizeof(int));
  auto d_right = MemoryAlloc(place, total_device * sizeof(int));
  int *d_left_ptr = reinterpret_cast<int *>(d_left->ptr());
  int *d_right_ptr = reinterpret_cast<int *>(d_right->ptr());

#if defined(PADDLE_WITH_CUDA)
  cudaMemsetAsync(d_left_ptr, -1, total_device * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_device * sizeof(int), stream);

#elif defined(PADDLE_WITH_XPU_KP)
  // get XPUContext according to xpu place
  phi::XPUContext xpu_dev_ctx(place);
  auto xpu_context = xpu_dev_ctx.x_context();

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      common::errors::External("XPU constant kernel return wrong value[%d %s]",
                               r,
                               XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(
      r2,
      XPU_SUCCESS,
      common::errors::External("XPU constant kernel return wrong value[%d %s]",
                               r2,
                               XPUAPIErrorMsg[r2]));
#endif

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t val_type_size = accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);
  VLOG(3) << "pull_sparse len:" << len << "  val_type_size: " << val_type_size;
  auto d_sorted_keys = MemoryAlloc(place, len * sizeof(KeyType));
  auto d_sorted_keys_ptr = reinterpret_cast<KeyType *>(d_sorted_keys->ptr());
  auto d_merged_keys = MemoryAlloc(place, len * sizeof(KeyType));
  auto d_merged_keys_ptr = reinterpret_cast<KeyType *>(d_merged_keys->ptr());
  auto d_restore_idx = MemoryAlloc(place, len * sizeof(uint32_t));
  auto d_restore_idx_ptr = reinterpret_cast<uint32_t *>(d_restore_idx->ptr());
  auto d_shard_keys = MemoryAlloc(place, len * sizeof(KeyType));
  auto d_shard_keys_ptr = reinterpret_cast<KeyType *>(d_shard_keys->ptr());
  auto d_shard_vals = MemoryAlloc(place, len * val_type_size);
  auto d_shard_vals_ptr = reinterpret_cast<float *>(d_shard_vals->ptr());

  size_t uniq_len = merge_keys(num,
                               d_keys,
                               len,
                               d_sorted_keys_ptr,
                               d_merged_keys_ptr,
                               d_restore_idx_ptr,
                               stream);
  sync_stream(stream);

  auto d_idx = MemoryAlloc(place, uniq_len * sizeof(int));
  auto d_idx_ptr = reinterpret_cast<int *>(d_idx->ptr());
  split_idx_to_shard(d_merged_keys_ptr,
                     d_idx_ptr,
                     uniq_len,
                     d_left_ptr,
                     d_right_ptr,
                     num,
                     stream);
  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, d_merged_keys_ptr, d_idx_ptr, uniq_len, stream, dev_id);

  auto dst_place = phi::CPUPlace();
  auto src_place = place;

  memory_copy(dst_place,
              h_left,
              src_place,
              d_left_ptr,
              total_device * sizeof(int),
              stream);
  memory_copy(dst_place,
              h_right,
              src_place,
              d_right_ptr,
              total_device * sizeof(int),
              stream);
  sync_stream(stream);

  if (!enable_gpu_direct_access_) {
    for (int i = 0; i < total_device; ++i) {
      int shard_len = h_right[i] - h_left[i] + 1;
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      create_storage(
          num, i, shard_len * sizeof(KeyType), shard_len * val_type_size);
    }
    walk_to_dest(num, total_device, h_left, h_right, d_shard_keys_ptr, NULL);
  }

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    auto &node = path_[num][i].nodes_.back();
    AnyDeviceGuard guard(resource_->dev_id(i));
    ptr_tables_[i]->rwlock_->RDLock();
    if (!enable_gpu_direct_access_) {
      ptr_tables_[i]->get(reinterpret_cast<KeyType *>(node.key_storage),
                          node.val_storage,
                          h_right[i] - h_left[i] + 1,
                          resource_->remote_stream(i, num),
                          gpu_accessor_);
    } else {
      ptr_tables_[i]->get(d_shard_keys_ptr + h_left[i],
                          reinterpret_cast<char *>(d_shard_vals_ptr) +
                              h_left[i] * val_type_size,
                          h_right[i] - h_left[i] + 1,
                          resource_->remote_stream(i, num),
                          gpu_accessor_);
    }
  }

  for (int i = 0; i < total_device; ++i) {
    AnyDeviceGuard guard(resource_->dev_id(i));
    sync_stream(resource_->remote_stream(i, num));
    if (h_left[i] == -1) {
      continue;
    }
    ptr_tables_[i]->rwlock_->UNLock();
  }

  if (!enable_gpu_direct_access_) {
    walk_to_src(num,
                total_device,
                h_left,
                h_right,
                reinterpret_cast<char *>(d_shard_vals_ptr),
                val_type_size);
  }

  AnyDeviceGuard guard2(dev_id);
  auto d_merged_vals = MemoryAlloc(place, uniq_len * val_type_size);
  auto d_merged_vals_ptr = reinterpret_cast<float *>(d_merged_vals->ptr());
  heter_comm_kernel_->dy_mf_fill_dvals(d_shard_vals_ptr,
                                       d_merged_vals_ptr,
                                       d_idx_ptr,
                                       uniq_len,
                                       val_type_size,
                                       stream);
  sync_stream(stream);

  heter_comm_kernel_->unpack_merged_vals(len,
                                         d_keys,
                                         d_merged_vals_ptr,
                                         d_restore_idx_ptr,
                                         d_vals,
                                         val_type_size,
                                         stream);
  sync_stream(stream);

  if (!enable_gpu_direct_access_) {
    for (int i = 0; i < total_device; ++i) {
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      destroy_storage(num, i);
    }
  }
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_normal_sparse(
    const int num, KeyType *d_keys, float *d_vals, size_t len) {
  int total_device = resource_->total_device();
  int dev_id = resource_->dev_id(num);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);

  int h_left[total_device];   // NOLINT
  int h_right[total_device];  // NOLINT

  auto d_left = MemoryAlloc(place, total_device * sizeof(int));
  auto d_right = MemoryAlloc(place, total_device * sizeof(int));
  int *d_left_ptr = reinterpret_cast<int *>(d_left->ptr());
  int *d_right_ptr = reinterpret_cast<int *>(d_right->ptr());

#if defined(PADDLE_WITH_CUDA)
  cudaMemsetAsync(d_left_ptr, -1, total_device * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_device * sizeof(int), stream);

#elif defined(PADDLE_WITH_XPU_KP)
  // get XPUContext according to xpu place
  phi::XPUContext xpu_dev_ctx(place);
  auto xpu_context = xpu_dev_ctx.x_context();

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      common::errors::External("XPU constant kernel return wrong value[%d %s]",
                               r,
                               XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(
      r2,
      XPU_SUCCESS,
      common::errors::External("XPU constant kernel return wrong value[%d %s]",
                               r2,
                               XPUAPIErrorMsg[r2]));
#endif

  auto d_idx = MemoryAlloc(place, len * sizeof(int));
  int *d_idx_ptr = reinterpret_cast<int *>(d_idx->ptr());

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t val_type_size = accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);
  VLOG(3) << "pull_sparse len:" << len << "  val_type_size: " << val_type_size;
  auto d_shard_keys = MemoryAlloc(place, len * sizeof(KeyType));
  KeyType *d_shard_keys_ptr = reinterpret_cast<KeyType *>(d_shard_keys->ptr());
  auto d_shard_vals = MemoryAlloc(place, len * val_type_size);
  float *d_shard_vals_ptr = reinterpret_cast<float *>(d_shard_vals->ptr());

  split_idx_to_shard(
      d_keys, d_idx_ptr, len, d_left_ptr, d_right_ptr, num, stream);

  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, d_keys, d_idx_ptr, len, stream, dev_id);

  auto dst_place = phi::CPUPlace();
  auto src_place = place;

  memory_copy(dst_place,
              h_left,
              src_place,
              d_left_ptr,
              total_device * sizeof(int),
              stream);
  memory_copy(dst_place,
              h_right,
              src_place,
              d_right_ptr,
              total_device * sizeof(int),
              stream);
  sync_stream(stream);

  if (!enable_gpu_direct_access_) {
    for (int i = 0; i < total_device; ++i) {
      int shard_len = h_right[i] - h_left[i] + 1;
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      create_storage(
          num, i, shard_len * sizeof(KeyType), shard_len * val_type_size);
    }
    walk_to_dest(num, total_device, h_left, h_right, d_shard_keys_ptr, NULL);
  }
  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    auto &node = path_[num][i].nodes_.back();
    AnyDeviceGuard guard(resource_->dev_id(i));
    ptr_tables_[i]->rwlock_->RDLock();
    if (!enable_gpu_direct_access_) {
      ptr_tables_[i]->get(reinterpret_cast<KeyType *>(node.key_storage),
                          node.val_storage,
                          h_right[i] - h_left[i] + 1,
                          resource_->remote_stream(i, num),
                          gpu_accessor_);
    } else {
      ptr_tables_[i]->get(d_shard_keys_ptr + h_left[i],
                          reinterpret_cast<char *>(d_shard_vals_ptr) +
                              h_left[i] * val_type_size,
                          h_right[i] - h_left[i] + 1,
                          resource_->remote_stream(i, num),
                          gpu_accessor_);
    }
  }

  for (int i = 0; i < total_device; ++i) {
    AnyDeviceGuard guard(resource_->dev_id(i));
    sync_stream(resource_->remote_stream(i, num));
    if (h_left[i] == -1) {
      continue;
    }
    ptr_tables_[i]->rwlock_->UNLock();
  }
  if (!enable_gpu_direct_access_) {
    walk_to_src(num,
                total_device,
                h_left,
                h_right,
                reinterpret_cast<char *>(d_shard_vals_ptr),
                val_type_size);
  }
  AnyDeviceGuard guard2(dev_id);
  heter_comm_kernel_->dy_mf_fill_dvals(
      d_shard_vals_ptr, d_vals, d_idx_ptr, len, val_type_size, stream);

  sync_stream(stream);

  if (!enable_gpu_direct_access_) {
    for (int i = 0; i < total_device; ++i) {
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      destroy_storage(num, i);
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_sparse(
    int num, KeyType *d_keys, float *d_vals, size_t len) {
  if (len == 0) {
    return;
  }
  if (multi_node_) {
    // infer graph split embedding by id hash mode
    if (FLAGS_graph_embedding_split_infer_mode && is_infer_mode_) {
      pull_normal_sparse(num, d_keys, d_vals, len);
    } else {
      pull_sparse_all2all(num, d_keys, d_vals, len);
    }
  } else {
    if (!FLAGS_gpugraph_dedup_pull_push_mode) {
      pull_merge_sparse(num, d_keys, d_vals, len);
    } else {
      pull_normal_sparse(num, d_keys, d_vals, len);
    }
  }
}

#if defined(PADDLE_WITH_CUDA)
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::push_sparse(
    int dev_num,
    KeyType *d_keys,
    float *d_grads,
    size_t len,
    Sgd &sgd) {  // NOLINT
  if (multi_node_) {
    push_sparse_all2all(dev_num, d_keys, d_grads, len, sgd);
  } else {
    push_normal_sparse(dev_num, d_keys, d_grads, len, sgd);
  }
  print_debug_time(dev_num);
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::push_normal_sparse(
    int dev_num,
    KeyType *d_keys,
    float *d_grads,
    size_t len,
    Sgd &sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int total_device = resource_->total_device();
  int dev_id = resource_->dev_id(dev_num);

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(dev_num, 0);

  int h_left[total_device];   // NOLINT
  int h_right[total_device];  // NOLINT

  auto d_left = MemoryAlloc(place, total_device * sizeof(int));
  auto d_right = MemoryAlloc(place, total_device * sizeof(int));
  int *d_left_ptr = reinterpret_cast<int *>(d_left->ptr());
  int *d_right_ptr = reinterpret_cast<int *>(d_right->ptr());

#if defined(PADDLE_WITH_CUDA)
  cudaMemsetAsync(d_left_ptr, -1, total_device * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_device * sizeof(int), stream);

#elif defined(PADDLE_WITH_XPU_KP)
  // get XPUContext according to xpu place
  phi::XPUContext xpu_dev_ctx(place);
  auto xpu_context = xpu_dev_ctx.x_context();

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      common::errors::External("XPU constant kernel return wrong value[%d %s]",
                               r,
                               XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(
      r2,
      XPU_SUCCESS,
      common::errors::External("XPU constant kernel return wrong value[%d %s]",
                               r2,
                               XPUAPIErrorMsg[r2]));
#endif

  auto d_idx = MemoryAlloc(place, len * sizeof(int));
  int *d_idx_ptr = reinterpret_cast<int *>(d_idx->ptr());

  auto d_shard_keys = MemoryAlloc(place, len * sizeof(KeyType));
  KeyType *d_shard_keys_ptr = reinterpret_cast<KeyType *>(d_shard_keys->ptr());

  float *d_shard_grads_ptr;
  auto d_shard_grads = MemoryAlloc(place, len * grad_value_size);
  d_shard_grads_ptr = reinterpret_cast<float *>(d_shard_grads->ptr());

  int uniq_len = len;
  if (!FLAGS_gpugraph_dedup_pull_push_mode) {
    size_t segment_len = 0;
    if (FLAGS_gpugraph_enable_segment_merge_grads) {
      // do two gradient merge
      // 1st. do segmented gradient merge
      // 2nd. do global gradient merge
      dynamic_merge_grad(
          dev_num, d_keys, d_grads, len, uniq_len, segment_len, true);
      len = segment_len;
      uniq_len = 0;
      segment_len = 0;
      dynamic_merge_grad(
          dev_num, d_keys, d_grads, len, uniq_len, segment_len, false);
    } else {
      // Perform gradient merge only once
      dynamic_merge_grad(
          dev_num, d_keys, d_grads, len, uniq_len, segment_len, false);
    }
  }

  split_idx_to_shard(
      d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr, dev_num, stream);

  heter_comm_kernel_->dy_mf_fill_shard_grads(d_shard_keys_ptr,
                                             d_keys,
                                             d_shard_grads_ptr,
                                             d_grads,
                                             d_idx_ptr,
                                             uniq_len,
                                             grad_value_size,
                                             stream,
                                             gpu_accessor_);

  auto dst_place = phi::CPUPlace();
  auto src_place = place;
  memory_copy(dst_place,
              h_left,
              src_place,
              d_left_ptr,
              total_device * sizeof(int),
              stream);
  memory_copy(dst_place,
              h_right,
              src_place,
              d_right_ptr,
              total_device * sizeof(int),
              stream);
  sync_stream(stream);

  if (!enable_gpu_direct_access_) {
    for (int i = 0; i < total_device; ++i) {
      int shard_len = h_right[i] - h_left[i] + 1;
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      create_storage(
          dev_num, i, shard_len * sizeof(KeyType), shard_len * grad_value_size);
    }

    walk_to_dest(dev_num,
                 total_device,
                 h_left,
                 h_right,
                 d_shard_keys_ptr,
                 reinterpret_cast<char *>(d_shard_grads_ptr),
                 grad_value_size);
  }

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto &node = path_[dev_num][i].nodes_.back();
    //    if (!enable_gpu_direct_access_) {
    //      sync_stream(node.in_stream);
    //    }

    AnyDeviceGuard guard(resource_->dev_id(i));
    ptr_tables_[i]->rwlock_->WRLock();
    if (!enable_gpu_direct_access_) {
      ptr_tables_[i]->update(reinterpret_cast<KeyType *>(node.key_storage),
                             node.val_storage,
                             h_right[i] - h_left[i] + 1,
                             sgd,
                             resource_->remote_stream(i, dev_num));
    } else {
      ptr_tables_[i]->update(d_shard_keys_ptr + h_left[i],
                             reinterpret_cast<char *>(d_shard_grads_ptr) +
                                 grad_value_size * h_left[i],
                             h_right[i] - h_left[i] + 1,
                             sgd,
                             resource_->remote_stream(i, dev_num));
    }
  }

  for (int i = 0; i < total_device; ++i) {
    AnyDeviceGuard guard(resource_->dev_id(i));
    sync_stream(resource_->remote_stream(i, dev_num));
    if (h_left[i] != -1) {
      ptr_tables_[i]->rwlock_->UNLock();
    }
  }

  if (!enable_gpu_direct_access_) {
    for (int i = 0; i < total_device; ++i) {
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      destroy_storage(dev_num, i);
    }
  }
}

#elif defined(PADDLE_WITH_XPU_KP)
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::push_sparse(
    int dev_num, KeyType *d_keys, GradType *d_grads, size_t len) {
  if (len == 0) {
    return;
  }

  int total_device = resource_->total_device();
  int dev_id = resource_->dev_id(dev_num);

  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(dev_num, 0);

  int h_left[total_device];   // NOLINT
  int h_right[total_device];  // NOLINT

  auto d_left = MemoryAlloc(place, total_device * sizeof(int));
  auto d_right = MemoryAlloc(place, total_device * sizeof(int));
  int *d_left_ptr = reinterpret_cast<int *>(d_left->ptr());
  int *d_right_ptr = reinterpret_cast<int *>(d_right->ptr());

#if defined(PADDLE_WITH_CUDA)
  cudaMemsetAsync(d_left_ptr, -1, total_device * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_device * sizeof(int), stream);

#elif defined(PADDLE_WITH_XPU_KP)
  // get XPUContext according to xpu place
  phi::XPUContext xpu_dev_ctx(place);
  auto xpu_context = xpu_dev_ctx.x_context();

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      common::errors::External("XPU constant kernel return wrong value[%d %s]",
                               r,
                               XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(
      r2,
      XPU_SUCCESS,
      common::errors::External("XPU constant kernel return wrong value[%d %s]",
                               r2,
                               XPUAPIErrorMsg[r2]));
#endif

  auto d_idx = MemoryAlloc(place, len * sizeof(int));
  int *d_idx_ptr = reinterpret_cast<int *>(d_idx->ptr());

  auto d_shard_keys = MemoryAlloc(place, len * sizeof(KeyType));
  KeyType *d_shard_keys_ptr = reinterpret_cast<KeyType *>(d_shard_keys->ptr());
  auto d_shard_grads = MemoryAlloc(place, len * sizeof(GradType));
  GradType *d_shard_grads_ptr =
      reinterpret_cast<GradType *>(d_shard_grads->ptr());

  int uniq_len = len;
  merge_grad(dev_num, d_keys, d_grads, len, uniq_len);

  split_idx_to_shard(
      d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr, dev_num, stream);

  heter_comm_kernel_->fill_shard_grads(d_shard_keys_ptr,
                                       d_keys,
                                       d_shard_grads_ptr,
                                       d_grads,
                                       d_idx_ptr,
                                       (int64_t)uniq_len,
                                       stream);

  auto dst_place = phi::CPUPlace();
  auto src_place = place;
  memory_copy(dst_place,
              h_left,
              src_place,
              d_left_ptr,
              total_device * sizeof(int),
              stream);
  memory_copy(dst_place,
              h_right,
              src_place,
              d_right_ptr,
              total_device * sizeof(int),
              stream);
  sync_stream(stream);

  for (int i = 0; i < total_device; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(
        dev_num, i, shard_len * sizeof(KeyType), shard_len * sizeof(GradType));
  }

  walk_to_dest(dev_num,
               total_device,
               h_left,
               h_right,
               d_shard_keys_ptr,
               d_shard_grads_ptr);

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto &node = path_[dev_num][i].nodes_.back();
    AnyDeviceGuard guard(resource_->dev_id(i));
    tables_[i]->rwlock_->WRLock();
    tables_[i]->update(reinterpret_cast<KeyType *>(node.key_storage),
                       reinterpret_cast<GradType *>(node.val_storage),
                       h_right[i] - h_left[i] + 1,
                       resource_->remote_stream(i, dev_num));
  }

  for (int i = 0; i < total_device; ++i) {
    AnyDeviceGuard guard(resource_->dev_id(i));
    sync_stream(resource_->remote_stream(i, dev_num));
    if (h_left[i] != -1) {
      tables_[i]->rwlock_->UNLock();
    }
  }

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    destroy_storage(dev_num, i);
  }
}

#endif

#if defined(PADDLE_WITH_CUDA)
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::update_one_table(
    int gpu_id,
    KeyType *d_keys,
    GradType *d_grads,
    size_t len,
    Sgd &sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int dev_id = resource_->dev_id(gpu_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_id, 0);
  // no mf dim
  if (!multi_mf_dim_) {
    auto &table = tables_[gpu_id];
    table->rwlock_->WRLock();
    table->update(
        d_keys, reinterpret_cast<const char *>(d_grads), len, sgd, stream);
    table->rwlock_->UNLock();
  } else {
    auto &table = ptr_tables_[gpu_id];
    table->rwlock_->WRLock();
    table->update(
        d_keys, reinterpret_cast<const char *>(d_grads), len, sgd, stream);
    table->rwlock_->UNLock();
  }
  cudaStreamSynchronize(stream);
}
#endif

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::end_pass() {
  int total_device = resource_->total_device();
  std::vector<std::thread> threads;

  auto dump_to_cpu_func = [this](int index) {
    auto stream = resource_->local_stream(index, 0);
    int dev_id = resource_->dev_id(index);
    AnyDeviceGuard guard(dev_id);
    tables_[index]->dump_to_cpu(dev_id, stream);
  };

  if (!multi_mf_dim_) {
    for (int i = 0; i < total_device; ++i) {
      threads.push_back(std::thread(dump_to_cpu_func, i));
    }
    for (auto &t : threads) {
      t.join();
    }
  }
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::dedup_keys_and_fillidx(
    const int gpu_id,
    const int total_fea_num,
    const KeyType *d_keys,   // input
    KeyType *d_merged_keys,  // output
    KeyType *d_sorted_keys,
    uint32_t *d_restore_idx,
    uint32_t *d_sorted_idx,
    uint32_t *d_offset,
    uint32_t *d_merged_cnts,
    bool filter_zero,
    cudaStream_t stream) {
  phi::GPUPlace place = phi::GPUPlace(gpu_id);
  platform::CUDADeviceGuard guard(gpu_id);
  if (stream == 0) {
    stream = resource_->local_stream(gpu_id, 0);
  }

  PADDLE_ENFORCE_GT(
      total_fea_num,
      0,
      common::errors::InvalidArgument(
          "Param total feature num should be greater than 0, but got %d.",
          total_fea_num));
  size_t merged_size = 0;
  size_t byte_size = sizeof(uint32_t) * (total_fea_num + 1);

  thread_local std::shared_ptr<memory::Allocation> d_index_ptr = nullptr;
  uint32_t *d_index_in = AllocCache<uint32_t>(&d_index_ptr, place, byte_size);
  int *d_merged_size = reinterpret_cast<int *>(&d_index_in[total_fea_num]);

  heter_comm_kernel_->fill_idx(d_index_in, total_fea_num, stream, gpu_id);

  void *d_buf = NULL;
  size_t temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(NULL,
                                      temp_storage_bytes,
                                      d_keys,
                                      d_sorted_keys,
                                      d_index_in,
                                      d_sorted_idx,
                                      total_fea_num,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream,
                                      false));
  thread_local std::shared_ptr<memory::Allocation> d_cache_ptr = nullptr;
  d_buf = AllocCache<void>(&d_cache_ptr, place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(d_buf,
                                      temp_storage_bytes,
                                      d_keys,
                                      d_sorted_keys,
                                      d_index_in,
                                      d_sorted_idx,
                                      total_fea_num,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream,
                                      false));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(NULL,
                                         temp_storage_bytes,
                                         d_sorted_keys,
                                         d_merged_keys,
                                         d_merged_cnts,
                                         d_merged_size,
                                         total_fea_num,
                                         stream));
  d_buf = AllocCache<void>(&d_cache_ptr, place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(d_buf,
                                         temp_storage_bytes,
                                         d_sorted_keys,
                                         d_merged_keys,
                                         d_merged_cnts,
                                         d_merged_size,
                                         total_fea_num,
                                         stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&merged_size,
                                             d_merged_size,
                                             sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, d_merged_cnts, d_offset, merged_size, stream));

  d_buf = AllocCache<void>(&d_cache_ptr, place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      d_buf, temp_storage_bytes, d_merged_cnts, d_offset, merged_size, stream));

  if (filter_zero) {
    cudaMemsetAsync(d_restore_idx, 0, total_fea_num * sizeof(uint32_t), stream);
  }
  // fill restore idx [1,3,5,2,4,6] = [1,2,1,3,2,1]
  heter_comm_kernel_->fill_restore_idx(filter_zero,
                                       total_fea_num,
                                       merged_size,
                                       d_merged_keys,
                                       d_sorted_idx,
                                       d_offset,
                                       d_merged_cnts,
                                       d_restore_idx,
                                       stream);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  return merged_size;
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_one_table(
    const int gpu_id,
    KeyType *d_keys,
    float *d_vals,
    const size_t &len,
    const cudaStream_t &stream) {
  // tracker need zero
  if (FLAGS_enable_tracker_all2all) {
    cudaMemsetAsync(d_vals, 0, len * pull_type_size_, stream);
  }

  ptr_tables_[gpu_id]->rwlock_->RDLock();
  ptr_tables_[gpu_id]->get(
      d_keys, reinterpret_cast<char *>(d_vals), len, stream, gpu_accessor_);
  ptr_tables_[gpu_id]->rwlock_->UNLock();

  // tracker
  if (FLAGS_enable_tracker_all2all) {
    // check pull values
    heter_comm_kernel_->check_valid_values(
        0,
        len,
        d_keys,
        reinterpret_cast<const char *>(d_vals),
        pull_type_size_,
        stream,
        (gpu_id == 0));
  }
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_sparse_all2all(
    const int &gpu_id, KeyType *d_keys, float *d_vals, const size_t &fea_num) {
  AnyDeviceGuard guard(gpu_id);
  auto &loc = storage_[gpu_id];
  // get from local table
  auto stream = resource_->local_stream(gpu_id, 0);

  size_t gather_inner_size = 0;
  size_t pull_size = 0;
  size_t value_bytes = pull_type_size_;
  loc.all2all_span_.Resume();
  // enable inner gather
  if (FLAGS_enable_sparse_inner_gather) {
    loc.inner_span_.Resume();
    // gather keys of all gpu and select shard key
    gather_inner_size =
        gather_inner_keys_by_copy(gpu_id, fea_num, d_keys, stream);
    loc.inner_span_.Pause();

    loc.node_span_.Resume();
    // all2all mode begins. init resource, partition keys, pull vals by all2all
    pull_size = gather_inter_keys_by_all2all(
        gpu_id, gather_inner_size, loc.d_merged_keys, stream);
    loc.node_span_.Pause();

    // pull one table
    pull_one_table(gpu_id,
                   loc.d_merged_keys,
                   reinterpret_cast<float *>(loc.d_merged_vals),
                   pull_size,
                   stream);

    // all2all
    loc.node_span_.Resume();
    // fp16
    if (FLAGS_enable_all2all_use_fp16) {
      value_bytes = heter_comm_kernel_->compress_values(
          pull_size,
          reinterpret_cast<const char *>(loc.d_merged_vals),
          reinterpret_cast<char *>(loc.d_merged_push_vals),
          pull_type_size_,
          max_mf_dim_,
          max_value_bound_,
          stream);

      scatter_inter_vals_by_all2all(gpu_id,
                                    gather_inner_size,
                                    loc.d_merged_push_vals,
                                    loc.d_merged_push_vals,
                                    value_bytes,
                                    loc.d_merged_vals,
                                    stream);
      // unzip fp16
      heter_comm_kernel_->uncompress_values(
          gather_inner_size,
          reinterpret_cast<const char *>(loc.d_merged_push_vals),
          reinterpret_cast<char *>(loc.d_merged_vals),
          pull_type_size_,
          max_mf_dim_,
          max_value_bound_,
          stream);

      // pull
      if (FLAGS_enable_tracker_all2all) {
        heter_comm_kernel_->check_valid_values(
            4,
            gather_inner_size,
            loc.d_merged_push_keys,
            reinterpret_cast<const char *>(loc.d_merged_vals),
            pull_type_size_,
            stream,
            (gpu_id == 0));
      }

      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    } else {
      scatter_inter_vals_by_all2all(gpu_id,
                                    gather_inner_size,
                                    loc.d_merged_vals,
                                    loc.d_merged_vals,
                                    pull_type_size_,
                                    loc.d_merged_push_vals,
                                    stream);
    }
    loc.node_span_.Pause();

    // innter scatter
    loc.inner_span_.Resume();
    scatter_inner_vals_by_copy(
        gpu_id, fea_num, loc.d_merged_vals, d_vals, pull_type_size_, stream);
    loc.inner_span_.Pause();
  } else {
    loc.alloc(fea_num, max_type_size_);
    loc.node_span_.Resume();
    // all2all mode begins. init resource, partition keys, pull vals by all2all
    pull_size =
        gather_inter_keys_by_all2all(gpu_id, fea_num, d_keys, stream, true);

    loc.node_span_.Pause();
    // get all tables
    pull_normal_sparse(gpu_id,
                       loc.d_merged_keys,
                       reinterpret_cast<float *>(loc.d_merged_vals),
                       pull_size);
    // all2all
    loc.node_span_.Resume();
    // fp16
    if (FLAGS_enable_all2all_use_fp16) {
      value_bytes = heter_comm_kernel_->compress_values(
          pull_size,
          reinterpret_cast<const char *>(loc.d_merged_vals),
          reinterpret_cast<char *>(loc.d_merged_push_vals),
          pull_type_size_,
          max_mf_dim_,
          max_value_bound_,
          stream);
      scatter_inter_vals_by_all2all(gpu_id,
                                    fea_num,
                                    loc.d_merged_push_vals,
                                    loc.d_merged_push_vals,
                                    value_bytes,
                                    loc.d_merged_vals,
                                    stream);
      heter_comm_kernel_->uncompress_values(
          gather_inner_size,
          reinterpret_cast<const char *>(loc.d_merged_push_vals),
          reinterpret_cast<char *>(loc.d_merged_vals),
          pull_type_size_,
          max_mf_dim_,
          max_value_bound_,
          stream);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    } else {
      scatter_inter_vals_by_all2all(gpu_id,
                                    fea_num,
                                    loc.d_merged_vals,
                                    d_vals,
                                    pull_type_size_,
                                    loc.d_merged_push_vals,
                                    stream);
    }
    loc.node_span_.Pause();
  }
  loc.all2all_span_.Pause();

  // pull
  if (FLAGS_enable_tracker_all2all) {
    heter_comm_kernel_->check_valid_values(
        1,
        fea_num,
        d_keys,
        reinterpret_cast<const char *>(d_vals),
        pull_type_size_,
        stream,
        (gpu_id == 0));
    VLOG(0) << "pull gpu id=" << gpu_id << ", fea num=" << fea_num
            << ", inner=" << gather_inner_size << ", node=" << pull_size
            << ", fp16=" << FLAGS_enable_all2all_use_fp16
            << ", compress=" << value_bytes
            << ", pull bytes=" << pull_type_size_;
  }
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::shard_inner_keys(
    const size_t &total_fea_num,
    const KeyType *d_keys,
    const int &gpu_id,
    const int &gpu_num,
    HeterCommType::InnerResource *res,
    const cudaStream_t &stream) {
  AnyDeviceGuard guard(gpu_id);
  thread_local std::vector<uint32_t> h_offsets;
  h_offsets.resize(gpu_num * 2);  // NOLINT
  uint32_t *d_left_ptr = res->d_offset_ptr;
  cudaMemsetAsync(d_left_ptr, -1, gpu_num * 2 * sizeof(int), stream);

  uint32_t *d_right_ptr = &d_left_ptr[gpu_num];
  split_idx_to_shard<uint32_t, cudaStream_t>(const_cast<KeyType *>(d_keys),
                                             res->d_idx,
                                             total_fea_num,
                                             d_left_ptr,
                                             d_right_ptr,
                                             gpu_id,
                                             stream);

  cudaMemcpyAsync(&h_offsets[0],
                  d_left_ptr,
                  gpu_num * 2 * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost,
                  stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  for (int i = 0; i < gpu_num; ++i) {
    uint32_t &h_right = h_offsets[gpu_num + i];
    uint32_t &h_left = h_offsets[i];
    if (static_cast<int>(h_right) == -1 || static_cast<int>(h_left) == -1) {
      res->h_part_sizes[i] = 0;
    } else {
      res->h_part_sizes[i] = h_right - h_left + 1;
    }
  }
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_inner_keys_p2p(
    const size_t &total_fea_num,
    const KeyType *d_keys,
    HeterCommType::InnerResource &res,  // NOLINT
    const int &gpu_id,
    const int &gpu_num,
    const int &trans_id,
    const cudaStream_t &stream) {
  AnyDeviceGuard guard(gpu_id);
  // gather all datas
  heter_comm_kernel_->gather_keys(
      res.d_keys_parted, d_keys, res.d_idx, total_fea_num, stream, gpu_id);
  if (trans_id < 0) {
    // not need transfer
    for (int i = 0; i < gpu_num; ++i) {
      size_t &data_len = res.h_part_sizes[i];
      if (data_len == 0) {
        continue;
      }
      size_t &offset = res.h_offsets[i];
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                     i,
                                                     &res.d_keys_parted[offset],
                                                     gpu_id,
                                                     data_len * sizeof(KeyType),
                                                     stream));
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    return;
  }
  // need transfer
  for (int i = 0; i < gpu_num; ++i) {
    size_t data_len = res.h_part_sizes[i];
    if (data_len == 0) {
      continue;
    }
    size_t &offset = res.h_offsets[i];
    // printf("[%d->%d->%d]send keys offset: %ld, len: %ld\n", gpu_id,
    // trans_id, i, offset, data_len);
    if (!need_transfer(gpu_id, i)) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                     i,
                                                     &res.d_keys_parted[offset],
                                                     gpu_id,
                                                     data_len * sizeof(KeyType),
                                                     stream));
      continue;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_trans_keys,
                                                   trans_id,
                                                   &res.d_keys_parted[offset],
                                                   gpu_id,
                                                   data_len * sizeof(KeyType),
                                                   stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                   i,
                                                   res.d_trans_keys,
                                                   trans_id,
                                                   data_len * sizeof(KeyType),
                                                   stream));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t
HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_inner_keys_by_copy(
    const int &gpu_id,
    const size_t &fea_size,
    const KeyType *d_keys,
    const cudaStream_t &stream) {
  auto &my_cache = storage_[gpu_id];
  auto &res = my_cache.inner_res;
  my_cache.init_inner(fea_size, device_num_);
  res.h_part_sizes = &my_cache.h_fea_sizes[0];
  // shard inner keys
  shard_inner_keys(fea_size, d_keys, gpu_id, device_num_, &res, stream);

  my_cache.inner_barrier_.Resume();
  // barrier wait all gpu done
  barrier_.wait();
  my_cache.inner_barrier_.Pause();

  size_t max_part_size = 0;
  size_t shard_recv_offset = 0;
  size_t shard_send_offset = 0;
  for (int i = 0; i < device_num_; ++i) {
    auto &cache = storage_[i];
    my_cache.h_recv_offsets[i] = shard_recv_offset;
    shard_recv_offset += cache.h_fea_sizes[gpu_id];
    res.h_offsets[i] = shard_send_offset;
    shard_send_offset += res.h_part_sizes[i];
    if (max_part_size < res.h_part_sizes[i]) {
      max_part_size = res.h_part_sizes[i];
    }
  }
  PADDLE_ENFORCE_EQ(
      shard_send_offset,
      static_cast<size_t>(fea_size),
      common::errors::InvalidArgument(
          "Param shard_send_offset should be equal to %d, but got %d.",
          static_cast<size_t>(fea_size),
          shard_send_offset));

  size_t trans_need_size =
      std::max(shard_recv_offset, static_cast<size_t>(fea_size));
  int trans_id = -1;
  if (topo_aware_ && device_num_ > 4) {
    trans_id = get_transfer_devid(gpu_id);
    storage_[trans_id].h_trans_size = max_part_size;
    // barrier wait all set trans length [0-4, 1-5, 3-7, 2-6]
    barrier_.wait();
    my_cache.h_trans_offset = trans_need_size;
    trans_need_size += my_cache.h_trans_size;
  }
  my_cache.alloc(trans_need_size, max_type_size_);

  my_cache.inner_barrier_.Resume();
  // barrier wait all hbm malloc size
  barrier_.wait();
  my_cache.inner_barrier_.Pause();

  for (int i = 0; i < device_num_; ++i) {
    auto &cache = storage_[i];
    size_t &recv_offset = cache.h_recv_offsets[gpu_id];
    res.d_remote_keys[i] = &cache.d_merged_keys[recv_offset];
    if (trans_id >= 0) {
      // set transfer buffer
      auto &trans_cache = storage_[trans_id];
      res.d_trans_keys = &trans_cache.d_merged_keys[trans_cache.h_trans_offset];
    }
  }
  res.d_keys_parted = my_cache.d_merged_push_keys;
  my_cache.inner_barrier_.Resume();
  // barrier wait set buffer ptr
  barrier_.wait();
  my_cache.inner_barrier_.Pause();
  gather_inner_keys_p2p(
      fea_size, d_keys, res, gpu_id, device_num_, trans_id, stream);
  // barrier wait all gpu aync memcpy data
  my_cache.inner_barrier_.Resume();
  barrier_.wait();
  my_cache.inner_barrier_.Pause();

  my_cache.init_pull(shard_recv_offset);

  size_t uniq_len = merge_keys(gpu_id,
                               my_cache.d_merged_keys,  // in keys
                               shard_recv_offset,
                               my_cache.d_merged_push_keys,  // sort keys
                               my_cache.d_merged_keys,       // out merge keys
                               my_cache.pull_res.d_restore_keys_idx,
                               stream);

  return uniq_len;
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::partition_shard_keys(
    const int &gpu_id,
    const size_t &len,
    const KeyType *d_keys,
    uint32_t *d_idx_parted,
    KeyType *d_keys_parted,
    size_t *h_part_sizes,
    const int &shard_num,
    const cudaStream_t &stream) {
  DevPlace place = DevPlace(gpu_id);
  AnyDeviceGuard guard(gpu_id);

  if (len <= 0) {
    for (int i = 0; i < shard_num; ++i) {
      h_part_sizes[i] = 0;
    }
    return;
  }

  thread_local std::shared_ptr<memory::Allocation> d_offset_tmp = nullptr;
  uint32_t *d_left = AllocCache<uint32_t>(
      &d_offset_tmp, place, (len * 3 + shard_num * 2) * sizeof(int));
  uint32_t *d_right = &d_left[shard_num];
  cudaMemsetAsync(d_left, -1, shard_num * 2 * sizeof(int), stream);

  uint32_t *d_idx_tmp_ptr = reinterpret_cast<uint32_t *>(&d_right[shard_num]);
  uint32_t *d_shard_index_ptr = &d_idx_tmp_ptr[len];
  uint32_t *d_shard_index_tmp_ptr = &d_shard_index_ptr[len];

  heter_comm_kernel_->fill_idx(d_idx_tmp_ptr, len, stream, gpu_id);
  if (resource_->keys2rank(gpu_id).get()) {
    // get dest rank by table
    // VLOG(0) << "cross sharding";
    resource_->keys2rank(gpu_id)->get_ranks(
        d_keys, d_shard_index_tmp_ptr, len, stream);
  } else {
    // get dest rank by sharding
    // VLOG(0) << "hard sharding";
    heter_comm_kernel_->calc_node_shard_index(
        d_keys, len, d_shard_index_tmp_ptr, device_num_, shard_num, stream);
  }

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(shard_num);
  heter_comm_kernel_->sort_pairs(NULL,
                                 temp_storage_bytes,
                                 d_shard_index_tmp_ptr,
                                 d_shard_index_ptr,
                                 d_idx_tmp_ptr,
                                 d_idx_parted,
                                 len,
                                 gpu_id,
                                 0,
                                 num_bits,
                                 stream);
  thread_local std::shared_ptr<memory::Allocation> d_temp_storage = nullptr;
  void *d_buf = AllocCache<void>(&d_temp_storage, place, temp_storage_bytes);
  heter_comm_kernel_->sort_pairs(d_buf,
                                 temp_storage_bytes,
                                 d_shard_index_tmp_ptr,
                                 d_shard_index_ptr,
                                 d_idx_tmp_ptr,
                                 d_idx_parted,
                                 len,
                                 gpu_id,
                                 0,
                                 num_bits,
                                 stream);
  heter_comm_kernel_->calc_shard_offset(
      d_shard_index_ptr, d_left, d_right, len, shard_num, stream, gpu_id);
  heter_comm_kernel_->gather_keys(
      d_keys_parted, d_keys, d_idx_parted, len, stream, gpu_id);

  thread_local std::vector<uint32_t> h_offsets;
  h_offsets.resize(shard_num * 2);
  cudaMemcpyAsync(&h_offsets[0],
                  d_left,
                  shard_num * 2 * sizeof(int),
                  cudaMemcpyDeviceToHost,
                  stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  for (int i = 0; i < shard_num; ++i) {
    uint32_t &h_right = h_offsets[shard_num + i];
    uint32_t &h_left = h_offsets[i];
    if (static_cast<int>(h_right) == -1 || static_cast<int>(h_left) == -1) {
      h_part_sizes[i] = 0;
    } else {
      h_part_sizes[i] = h_right - h_left + 1;
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::send_data_by_all2all(
    const int &gpu_id,
    const int &nccl_node_size,
    const int &nccl_rank_id,
    const int &value_bytes,
    const size_t *h_send_part_sizes,
    const size_t *h_send_part_offsets,
    const size_t *h_recv_part_sizes,
    const size_t *h_recv_part_offsets,
    const char *d_send_buff,
    char *d_rev_buff,
    const cudaStream_t &stream) {
  AnyDeviceGuard guard(resource_->dev_id(gpu_id));
  auto &comm = nccl_inter_comms_[gpu_id];
  const size_t &send_size = h_send_part_sizes[nccl_rank_id];
  size_t send_offset = h_send_part_offsets[nccl_rank_id] * value_bytes;
  size_t recv_offset = h_recv_part_offsets[nccl_rank_id] * value_bytes;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      reinterpret_cast<void *>(&d_rev_buff[recv_offset]),  // output
      &d_send_buff[send_offset],
      send_size * value_bytes,
      cudaMemcpyDeviceToDevice,
      stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  PADDLE_ENFORCE_EQ(send_size,
                    h_recv_part_sizes[nccl_rank_id],
                    common::errors::InvalidArgument(
                        "Param send_size should be equal to %d, but got %d.",
                        h_recv_part_sizes[nccl_rank_id],
                        send_size));

  auto &loc = storage_[gpu_id];
  auto nccl_stream = resource_->comm_stream(gpu_id, 0);
  size_t total_fea_num = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGroupStart());
  for (int i = 0; i < nccl_node_size; i++) {
    if (i == nccl_rank_id) {
      continue;
    }
    const size_t &send_size = h_send_part_sizes[i];
    if (send_size > 0) {
      send_offset = h_send_part_offsets[i] * value_bytes;
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::ncclSend(&d_send_buff[send_offset],
                                 send_size * value_bytes,
                                 ncclInt8,
                                 i,
                                 comm,
                                 nccl_stream));
      total_fea_num += send_size;
    }
    const size_t &recv_size = h_recv_part_sizes[i];
    if (recv_size > 0) {
      recv_offset = h_recv_part_offsets[i] * value_bytes;
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRecv(
          reinterpret_cast<void *>(&d_rev_buff[recv_offset]),
          recv_size * value_bytes,
          ncclInt8,
          i,
          comm,
          nccl_stream));
      total_fea_num += recv_size;
    }
  }
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(nccl_stream));

  return total_fea_num;
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    gather_inter_keys_by_all2all(const int &gpu_id,
                                 const size_t &fea_size,
                                 const KeyType *d_in_keys,
                                 const cudaStream_t &stream,
                                 bool debug) {
  AnyDeviceGuard guard(resource_->dev_id(gpu_id));
  auto &cache = storage_[gpu_id];
  cache.init_shard(fea_size, node_size_);
  auto &res = cache.shard_res;
  cache.total_keys_ += fea_size;

  size_t *h_local_part_sizes = res.h_local_part_sizes.data();
  size_t *h_local_part_offsets = res.h_local_part_offsets.data();
  uint32_t *h_push_fea_sizes = res.h_push_fea_sizes.data();
  // partition keys
  partition_shard_keys(gpu_id,
                       fea_size,
                       d_in_keys,
                       res.d_local_idx_parted,
                       cache.d_merged_push_keys,
                       h_local_part_sizes,
                       node_size_,
                       stream);

  int all_shard_part_size = node_size_ * node_size_;
  int rank_offset = rank_id_ * node_size_;
  h_local_part_offsets[0] = 0;
  for (int i = 0; i < node_size_; ++i) {
    h_push_fea_sizes[rank_offset + i] = h_local_part_sizes[i];
    h_local_part_offsets[i + 1] =
        h_local_part_offsets[i] + h_local_part_sizes[i];
    if (i == rank_id_) {
      cache.local_keys_ += h_local_part_sizes[rank_id_];
    } else {
      cache.remote_keys_ += h_local_part_sizes[i];
    }
  }
  PADDLE_ENFORCE_EQ(fea_size,
                    h_local_part_offsets[node_size_],
                    common::errors::InvalidArgument(
                        "Param fea_size should be equal to %d, but got %d.",
                        h_local_part_offsets[node_size_],
                        fea_size));

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&res.d_node_size_ptr[rank_offset],
                                             &h_push_fea_sizes[rank_offset],
                                             node_size_ * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  cache.node_barrier_.Resume();
  auto &comm = nccl_inter_comms_[gpu_id];
  auto nccl_stream = resource_->comm_stream(gpu_id, 0);
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclAllGather(&res.d_node_size_ptr[rank_offset],
                                  reinterpret_cast<void *>(res.d_node_size_ptr),
                                  node_size_,
                                  ncclInt,
                                  comm,
                                  nccl_stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(nccl_stream));
  cache.node_barrier_.Pause();

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&h_push_fea_sizes[0],
                                             res.d_node_size_ptr,
                                             all_shard_part_size * sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  size_t *h_remote_part_sizes = res.h_remote_part_sizes.data();
  size_t *h_remote_part_offsets = res.h_remote_part_offsets.data();
  h_remote_part_offsets[0] = 0;
  for (int i = 0; i < node_size_; i++) {
    int offset = node_size_ * i + rank_id_;
    h_remote_part_sizes[i] = h_push_fea_sizes[offset];
    h_remote_part_offsets[i + 1] =
        h_remote_part_offsets[i] + h_remote_part_sizes[i];
  }
  size_t &remote_size = h_remote_part_offsets[node_size_];
  cache.alloc(remote_size, max_type_size_, HeterCommType::COPY_KEY);

  size_t total_fea_num = 0;
  if (rdma_checker_->need_rdma_trans()) {
    total_fea_num = send_keys_by_all2all_trans(gpu_id,
                                               rank_id_,
                                               node_size_,
                                               fea_size,
                                               cache.d_merged_push_keys,
                                               cache.d_merged_keys,
                                               stream);
  } else {
    cache.node_trans_.Resume();
    total_fea_num = send_data_by_all2all(
        gpu_id,
        node_size_,
        rank_id_,
        sizeof(KeyType),
        h_local_part_sizes,
        h_local_part_offsets,
        h_remote_part_sizes,
        h_remote_part_offsets,
        reinterpret_cast<const char *>(cache.d_merged_push_keys),
        reinterpret_cast<char *>(cache.d_merged_keys),
        stream);
    cache.node_trans_.Pause();
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  return remote_size;
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    scatter_inter_vals_by_all2all(const int &gpu_id,
                                  const size_t &fea_size,
                                  const char *d_in_vals,
                                  void *d_out_vals,
                                  const size_t &value_bytes,
                                  void *d_tmp_vals,
                                  const cudaStream_t &stream) {
  AnyDeviceGuard guard(resource_->dev_id(gpu_id));
  auto &cache = storage_[gpu_id];
  auto &res = cache.shard_res;
  auto h_local_part_sizes = res.h_local_part_sizes.data();
  auto h_local_part_offsets = res.h_local_part_offsets.data();
  auto h_remote_part_sizes = res.h_remote_part_sizes.data();
  auto h_remote_part_offsets = res.h_remote_part_offsets.data();

  size_t total_fea_num = 0;
  if (rdma_checker_->need_rdma_trans()) {
    total_fea_num =
        send_vals_by_all2all_trans(gpu_id,
                                   rank_id_,
                                   node_size_,
                                   d_in_vals,
                                   reinterpret_cast<char *>(d_tmp_vals),
                                   value_bytes,
                                   stream);
  } else {
    // send local device
    cache.node_trans_.Resume();
    total_fea_num = send_data_by_all2all(gpu_id,
                                         node_size_,
                                         rank_id_,
                                         value_bytes,
                                         h_remote_part_sizes,
                                         h_remote_part_offsets,
                                         h_local_part_sizes,
                                         h_local_part_offsets,
                                         d_in_vals,
                                         reinterpret_cast<char *>(d_tmp_vals),
                                         stream);
    cache.node_trans_.Pause();
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  // fill vals
  heter_comm_kernel_->scatter_vals(
      reinterpret_cast<const float *>(d_tmp_vals),  // in
      reinterpret_cast<float *>(d_out_vals),        // out
      res.d_local_idx_parted,
      fea_size,
      value_bytes,
      stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    recalc_local_and_remote_size(const int &gpu_id,
                                 const size_t &pull_size,
                                 const size_t &node_num,
                                 const uint32_t *d_tmp_size_list,
                                 const uint32_t *d_inter_size_list,
                                 const cudaStream_t &stream) {
  AnyDeviceGuard guard(resource_->dev_id(gpu_id));
  auto &cache = storage_[gpu_id];
  auto &res = cache.shard_res;
  auto h_local_part_sizes = res.h_local_part_sizes.data();
  auto h_local_part_offsets = res.h_local_part_offsets.data();
  auto h_remote_part_sizes = res.h_remote_part_sizes.data();
  auto h_remote_part_offsets = res.h_remote_part_offsets.data();

  std::vector<uint32_t> h_before_scatter_size_list(pull_size, 0);
  std::vector<uint32_t> h_end_scatter_size_list(node_num, 0);
  CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<char *>(h_before_scatter_size_list.data()),
      d_tmp_size_list,
      sizeof(uint32_t) * pull_size,
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(
      cudaMemcpyAsync(reinterpret_cast<char *>(h_end_scatter_size_list.data()),
                      d_inter_size_list,
                      sizeof(uint32_t) * node_num,
                      cudaMemcpyDeviceToHost,
                      stream));
  std::vector<size_t> vari_local_part_sizes(node_size_, 0);
  std::vector<size_t> vari_local_part_offsets(node_size_ + 1, 0);
  std::vector<size_t> vari_remote_part_sizes(node_size_, 0);
  std::vector<size_t> vari_remote_part_offsets(node_size_ + 1, 0);

  // local use end scatter(len is node num), reote use before scatter(len is
  // pull size) recompute offsets and parts
  VLOG(2) << "begin recalc local and remote size and offsets";
  for (int i = 0; i < node_size_; i++) {
    size_t local_size = 0;
    size_t remote_size = 0;
    for (int j = h_local_part_offsets[i]; j < h_local_part_offsets[i + 1];
         j++) {
      local_size += h_end_scatter_size_list[j];
    }
    vari_local_part_sizes[i] = local_size;
    vari_local_part_offsets[i + 1] =
        vari_local_part_offsets[i] + vari_local_part_sizes[i];
    VLOG(2) << "gpu id: " << gpu_id
            << ", before calc, local size:" << h_local_part_sizes[i]
            << ", local offset: " << h_local_part_offsets[i + 1]
            << ", end calc, local part size:" << vari_local_part_sizes[i]
            << ", local offsets: " << vari_local_part_offsets[i + 1];

    for (int k = h_remote_part_offsets[i]; k < h_remote_part_offsets[i + 1];
         k++) {
      remote_size += h_before_scatter_size_list[k];
    }
    vari_remote_part_sizes[i] = remote_size;
    vari_remote_part_offsets[i + 1] =
        vari_remote_part_offsets[i] + vari_remote_part_sizes[i];
    VLOG(2) << "gpu id: " << gpu_id
            << ", before cal, remote size:" << h_remote_part_sizes[i]
            << ", remote offset: " << h_remote_part_offsets[i + 1]
            << ", end calc, remote part size: " << vari_remote_part_sizes[i]
            << ", remote offsets: " << vari_remote_part_offsets[i + 1];
  }
  VLOG(2) << "end recalc remote size and offsets";
  // send 前把当前gpuid 的size 和 offset 替换成 vari size 和offset
  res.h_local_part_sizes = std::move(vari_local_part_sizes);
  res.h_local_part_offsets = std::move(vari_local_part_offsets);
  res.h_remote_part_sizes = std::move(vari_remote_part_sizes);
  res.h_remote_part_offsets = std::move(vari_remote_part_offsets);
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::scatter_inner_vals_p2p(
    const size_t &total_fea_num,
    void *d_out_vals,
    HeterCommType::InnerResource &res,  // NOLINT
    const int &gpu_id,
    const int &gpu_num,
    const int &trans_id,
    const size_t &value_bytes,
    const cudaStream_t &stream) {
  AnyDeviceGuard guard(resource_->dev_id(gpu_id));
  if (trans_id < 0) {
    // not need transfer
    for (int i = 0; i < gpu_num; ++i) {
      size_t &data_len = res.h_part_sizes[i];
      if (data_len == 0) {
        continue;
      }
      size_t &offset = res.h_offsets[i];
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpyPeerAsync(&res.d_vals_parted[offset * value_bytes],
                              gpu_id,
                              res.d_remote_vals[i],
                              i,
                              data_len * value_bytes,
                              stream));
    }
  } else {
    // need transfer
    for (int i = 0; i < gpu_num; ++i) {
      size_t data_len = res.h_part_sizes[i];
      if (data_len == 0) {
        continue;
      }
      size_t &offset = res.h_offsets[i];
      // printf("[%d<-%d<-%d]recv vals offset: %ld, len: %ld\n", gpu_id,
      // trans_id, i, offset, data_len);
      if (!need_transfer(gpu_id, i)) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpyPeerAsync(&res.d_vals_parted[offset * value_bytes],
                                gpu_id,
                                res.d_remote_vals[i],
                                i,
                                data_len * value_bytes,
                                stream));
        continue;
      }
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_trans_vals,
                                                     trans_id,
                                                     res.d_remote_vals[i],
                                                     i,
                                                     data_len * value_bytes,
                                                     stream));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpyPeerAsync(&res.d_vals_parted[offset * value_bytes],
                              gpu_id,
                              res.d_trans_vals,
                              trans_id,
                              data_len * value_bytes,
                              stream));
    }
  }
  // restore vals
  heter_comm_kernel_->scatter_vals(
      reinterpret_cast<const float *>(res.d_vals_parted),  // in
      reinterpret_cast<float *>(d_out_vals),               // out
      res.d_idx,
      total_fea_num,
      value_bytes,
      stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    scatter_inner_vals_by_copy(const int &gpu_id,
                               const size_t &fea_size,
                               const char *d_in_vals,
                               void *d_out_vals,
                               const size_t &value_bytes,
                               const cudaStream_t &stream) {
  AnyDeviceGuard guard(resource_->dev_id(gpu_id));
  auto &my_cache = storage_[gpu_id];
  // restore vals
  heter_comm_kernel_->gather_vals(
      reinterpret_cast<float *>(my_cache.d_merged_push_vals),  // out
      reinterpret_cast<const float *>(d_in_vals),              // in
      my_cache.pull_res.d_restore_keys_idx,
      my_cache.pull_res.h_recv_fea_num,
      value_bytes,
      stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  auto &res = my_cache.inner_res;
  int trans_id = -1;
  if (topo_aware_ && device_num_ > 4) {
    trans_id = get_transfer_devid(gpu_id);
  }
  my_cache.inner_barrier_.Resume();
  // barrier wait set buffer ptr
  barrier_.wait();
  my_cache.inner_barrier_.Pause();

  for (int i = 0; i < device_num_; ++i) {
    auto &cache = storage_[i];
    size_t &recv_offset = cache.h_recv_offsets[gpu_id];
    res.d_remote_vals[i] = &cache.d_merged_push_vals[recv_offset * value_bytes];
    if (trans_id >= 0) {
      // set transfer buffer
      auto &trans_cache = storage_[trans_id];
      res.d_trans_vals =
          &trans_cache
               .d_merged_push_vals[trans_cache.h_trans_offset * value_bytes];
    }
  }
  res.d_vals_parted = my_cache.d_merged_vals;
  my_cache.inner_barrier_.Resume();
  // barrier wait set buffer ptr
  barrier_.wait();
  my_cache.inner_barrier_.Pause();
  // recv all pull sparse vals
  scatter_inner_vals_p2p(fea_size,
                         d_out_vals,  // out
                         res,
                         gpu_id,
                         device_num_,
                         trans_id,
                         value_bytes,
                         stream);
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_inner_data_p2p(
    const size_t &total_fea_num,
    const KeyType *d_keys,
    const void *d_vals,
    HeterCommType::InnerResource &res,  // NOLINT
    const int &gpu_id,
    const int &gpu_num,
    const int &trans_id,
    const size_t &value_bytes,
    const cudaStream_t &stream) {
  AnyDeviceGuard guard(resource_->dev_id(gpu_id));
  // gather all datas
  heter_comm_kernel_->gather_keys(
      res.d_keys_parted, d_keys, res.d_idx, total_fea_num, stream, gpu_id);
  heter_comm_kernel_->gather_vals(reinterpret_cast<float *>(res.d_vals_parted),
                                  reinterpret_cast<const float *>(d_vals),
                                  res.d_idx,
                                  total_fea_num,
                                  value_bytes,
                                  stream);
  // p2p copy key and values
  if (trans_id < 0) {
    // not need transfer
    for (int i = 0; i < gpu_num; ++i) {
      size_t &data_len = res.h_part_sizes[i];
      if (data_len == 0) {
        continue;
      }
      size_t &offset = res.h_offsets[i];
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                     i,
                                                     &res.d_keys_parted[offset],
                                                     gpu_id,
                                                     data_len * sizeof(KeyType),
                                                     stream));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpyPeerAsync(res.d_remote_vals[i],
                              i,
                              &res.d_vals_parted[offset * value_bytes],
                              gpu_id,
                              data_len * value_bytes,
                              stream));
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    return;
  }
  // need transfer
  for (int i = 0; i < gpu_num; ++i) {
    size_t data_len = res.h_part_sizes[i];
    if (data_len == 0) {
      continue;
    }
    size_t &offset = res.h_offsets[i];
    if (!need_transfer(gpu_id, i)) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                     i,
                                                     &res.d_keys_parted[offset],
                                                     gpu_id,
                                                     data_len * sizeof(KeyType),
                                                     stream));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpyPeerAsync(res.d_remote_vals[i],
                              i,
                              &res.d_vals_parted[offset * value_bytes],
                              gpu_id,
                              data_len * value_bytes,
                              stream));
      continue;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_trans_keys,
                                                   trans_id,
                                                   &res.d_keys_parted[offset],
                                                   gpu_id,
                                                   data_len * sizeof(KeyType),
                                                   stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                   i,
                                                   res.d_trans_keys,
                                                   trans_id,
                                                   data_len * sizeof(KeyType),
                                                   stream));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyPeerAsync(res.d_trans_vals,
                            trans_id,
                            &res.d_vals_parted[offset * value_bytes],
                            gpu_id,
                            data_len * value_bytes,
                            stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_vals[i],
                                                   i,
                                                   res.d_trans_vals,
                                                   trans_id,
                                                   data_len * value_bytes,
                                                   stream));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::push_sparse_all2all(
    const int &gpu_id,
    KeyType *d_keys,
    float *d_grads,
    const size_t &len,
    Sgd &sgd) {  // NOLINT
  if (len == 0) {
    return;
  }
  AnyDeviceGuard guard(resource_->dev_id(gpu_id));
  auto &my_cache = storage_[gpu_id];
  my_cache.all2all_span_.Resume();
  auto stream = resource_->local_stream(gpu_id, 0);
  // tracker
  if (FLAGS_enable_tracker_all2all) {
    // check push grads
    heter_comm_kernel_->check_valid_values(
        10,
        len,
        d_keys,
        reinterpret_cast<const char *>(d_grads),
        grad_type_size_,
        stream,
        (gpu_id == 0));
  }
  // scale grad
  heter_comm_kernel_->scale_grad(len,
                                 reinterpret_cast<char *>(d_grads),
                                 grad_type_size_,
                                 max_mf_dim_,
                                 stream,
                                 gpu_accessor_);

  size_t inter_push_len = 0;
  size_t node_push_len = 0;
  size_t value_bytes = grad_type_size_;
  // enable inner gather
  if (FLAGS_enable_sparse_inner_gather) {
    my_cache.inner_span_.Resume();
    inter_push_len =
        gather_inner_gradient_by_copy(gpu_id,
                                      len,
                                      d_keys,
                                      reinterpret_cast<void *>(d_grads),
                                      grad_type_size_,
                                      stream);
    my_cache.inner_span_.Pause();

    my_cache.node_span_.Resume();

    if (FLAGS_enable_all2all_use_fp16) {  // use fp16
      value_bytes = heter_comm_kernel_->compress_values(
          inter_push_len,
          reinterpret_cast<const char *>(my_cache.d_merged_push_vals),
          reinterpret_cast<char *>(my_cache.d_merged_vals),
          grad_type_size_,
          max_mf_dim_,
          max_grad_bound_,
          stream);
      node_push_len =
          gather_sparse_gradient_by_all2all(gpu_id,
                                            inter_push_len,
                                            my_cache.d_merged_push_keys,
                                            my_cache.d_merged_vals,
                                            value_bytes,
                                            my_cache.d_merged_push_keys,
                                            my_cache.d_merged_keys,
                                            my_cache.d_merged_vals,
                                            my_cache.d_merged_push_vals,
                                            stream);
      heter_comm_kernel_->uncompress_values(
          node_push_len,
          reinterpret_cast<const char *>(my_cache.d_merged_vals),
          reinterpret_cast<char *>(my_cache.d_merged_push_vals),
          grad_type_size_,
          max_mf_dim_,
          max_grad_bound_,
          stream);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    } else {
      node_push_len =
          gather_sparse_gradient_by_all2all(gpu_id,
                                            inter_push_len,
                                            my_cache.d_merged_push_keys,
                                            my_cache.d_merged_push_vals,
                                            value_bytes,
                                            my_cache.d_merged_push_keys,
                                            my_cache.d_merged_keys,
                                            my_cache.d_merged_push_vals,
                                            my_cache.d_merged_vals,
                                            stream);
    }
    my_cache.node_span_.Pause();
  } else {  // only node all2all
    my_cache.node_span_.Resume();
    barrier_.wait();
    if (FLAGS_enable_all2all_use_fp16) {  // use fp16
      value_bytes = heter_comm_kernel_->compress_values(
          len,
          reinterpret_cast<const char *>(d_grads),
          reinterpret_cast<char *>(my_cache.d_merged_vals),
          grad_type_size_,
          max_mf_dim_,
          max_grad_bound_,
          stream);
      node_push_len =
          gather_sparse_gradient_by_all2all(gpu_id,
                                            len,
                                            d_keys,
                                            my_cache.d_merged_vals,
                                            value_bytes,
                                            my_cache.d_merged_push_keys,
                                            my_cache.d_merged_keys,
                                            my_cache.d_merged_vals,
                                            my_cache.d_merged_push_vals,
                                            stream);
      heter_comm_kernel_->uncompress_values(
          node_push_len,
          reinterpret_cast<const char *>(my_cache.d_merged_vals),
          reinterpret_cast<char *>(my_cache.d_merged_push_vals),
          grad_type_size_,
          max_mf_dim_,
          max_grad_bound_,
          stream);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    } else {
      node_push_len = gather_sparse_gradient_by_all2all(
          gpu_id,
          len,
          d_keys,                                   // in
          reinterpret_cast<const char *>(d_grads),  // in
          value_bytes,
          my_cache.d_merged_push_keys,  // out
          my_cache.d_merged_keys,       // tmp
          my_cache.d_merged_push_vals,  // out
          my_cache.d_merged_vals,       // tmp
          stream);
    }
    my_cache.node_span_.Pause();
  }
  if (FLAGS_enable_tracker_all2all) {
    VLOG(0) << "push gpu id=" << gpu_id
            << ", gather_sparse_gradient_by_all2all len=" << node_push_len;
  }
  // all embedx merge
  size_t uniq_len = merge_grad(gpu_id,
                               node_push_len,
                               my_cache.d_merged_push_keys,  // in
                               my_cache.d_merged_keys,       // out
                               my_cache.d_merged_push_vals,  // in
                               my_cache.d_merged_vals,
                               stream);  // out
  if (FLAGS_enable_tracker_all2all) {
    // check all2ll merge grads
    heter_comm_kernel_->check_valid_values(
        11,
        uniq_len,
        my_cache.d_merged_keys,
        reinterpret_cast<const char *>(my_cache.d_merged_vals),
        grad_type_size_,
        stream,
        (gpu_id == 0));
  }
  if (FLAGS_enable_sparse_inner_gather) {
    // update all grad
    update_one_table(gpu_id,
                     my_cache.d_merged_keys,
                     reinterpret_cast<GradType *>(my_cache.d_merged_vals),
                     uniq_len,
                     sgd);
  } else {
    // update all tables
    push_normal_sparse(gpu_id,
                       my_cache.d_merged_keys,
                       reinterpret_cast<float *>(my_cache.d_merged_vals),
                       uniq_len,
                       sgd);
  }
  my_cache.all2all_span_.Pause();
  // push
  if (FLAGS_enable_tracker_all2all) {
    VLOG(0) << "push gpu id=" << gpu_id << ", push len=" << len
            << ", inner=" << inter_push_len << ", node=" << node_push_len
            << ", update=" << uniq_len << ", compress bytes=" << value_bytes
            << ", grad_type_size=" << grad_type_size_;
  }
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::merge_grad(
    const int &gpu_id,
    const size_t &len,
    const KeyType *d_in_keys,
    KeyType *d_out_keys,
    const void *d_in_grads,
    void *d_out_grads,
    const cudaStream_t &stream) {
  platform::CUDADeviceGuard guard(gpu_id);
  auto place = phi::GPUPlace(gpu_id);
  thread_local std::shared_ptr<memory::Allocation> d_fea_num_info = nullptr;
  uint32_t *d_offset =
      AllocCache<uint32_t>(&d_fea_num_info, place, sizeof(uint32_t) * len * 4);
  uint32_t *d_sorted_idx = &d_offset[len];
  uint32_t *d_restore_idx = &d_sorted_idx[len];
  uint32_t *d_merged_cnts = &d_restore_idx[len];

  thread_local std::shared_ptr<memory::Allocation> d_sort_keys_ptr = nullptr;
  KeyType *d_sorted_keys =
      AllocCache<KeyType>(&d_sort_keys_ptr, place, sizeof(KeyType) * len);

  size_t merge_size = dedup_keys_and_fillidx(gpu_id,
                                             len,
                                             d_in_keys,   // input
                                             d_out_keys,  // output
                                             d_sorted_keys,
                                             d_restore_idx,
                                             d_sorted_idx,
                                             d_offset,
                                             d_merged_cnts,
                                             false,
                                             stream);

  heter_comm_kernel_->merge_gradient(d_out_keys,
                                     d_offset,
                                     d_merged_cnts,
                                     d_sorted_idx,
                                     reinterpret_cast<const char *>(d_in_grads),
                                     reinterpret_cast<char *>(d_out_grads),
                                     static_cast<int>(merge_size),
                                     max_mf_dim_,
                                     grad_type_size_,
                                     merger_,
                                     stream,
                                     gpu_accessor_);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  return merge_size;
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    gather_inner_gradient_by_copy(const int &gpu_id,
                                  const size_t &push_size,
                                  KeyType *d_keys,
                                  void *d_push_vals,
                                  const size_t &value_bytes,
                                  const cudaStream_t &stream) {
  auto &my_cache = storage_[gpu_id];

  my_cache.init_inner(push_size, device_num_);
  auto &res = my_cache.inner_res;
  res.h_part_sizes = &my_cache.h_fea_sizes[0];
  // shard data
  shard_inner_keys(push_size, d_keys, gpu_id, device_num_, &res, stream);
  my_cache.inner_barrier_.Resume();
  // barrier wait all gpu done
  barrier_.wait();
  my_cache.inner_barrier_.Pause();

  size_t max_part_size = 0;
  size_t shard_recv_offset = 0;
  size_t shard_send_offset = 0;
  for (int i = 0; i < device_num_; ++i) {
    auto &cache = storage_[i];
    my_cache.h_recv_offsets[i] = shard_recv_offset;
    shard_recv_offset += cache.h_fea_sizes[gpu_id];
    res.h_offsets[i] = shard_send_offset;
    shard_send_offset += res.h_part_sizes[i];
    if (res.h_part_sizes[i] > max_part_size) {
      max_part_size = res.h_part_sizes[i];
    }
  }

  size_t trans_need_size = std::max(shard_recv_offset, push_size);
  int trans_id = -1;
  if (topo_aware_ && device_num_ > 4) {
    trans_id = get_transfer_devid(gpu_id);
    storage_[trans_id].h_trans_size = max_part_size;
    // barrier wait all set trans length [0-4, 1-5, 3-7, 2-6]
    barrier_.wait();
    my_cache.h_trans_offset = trans_need_size;
    trans_need_size += my_cache.h_trans_size;
  }
  my_cache.alloc(trans_need_size, max_type_size_);
  my_cache.inner_barrier_.Resume();
  // barrier wait all hbm malloc size
  barrier_.wait();
  my_cache.inner_barrier_.Pause();

  for (int i = 0; i < device_num_; ++i) {
    auto &cache = storage_[i];
    size_t &recv_offset = cache.h_recv_offsets[gpu_id];
    res.d_remote_keys[i] = &cache.d_merged_keys[recv_offset];
    res.d_remote_vals[i] = &cache.d_merged_vals[recv_offset * value_bytes];
    if (trans_id >= 0) {
      // set transfer buffer
      auto &trans_cache = storage_[trans_id];
      res.d_trans_keys = &trans_cache.d_merged_keys[trans_cache.h_trans_offset];
      res.d_trans_vals =
          &trans_cache.d_merged_vals[trans_cache.h_trans_offset * value_bytes];
    }
  }
  res.d_keys_parted = my_cache.d_merged_push_keys;
  res.d_vals_parted = my_cache.d_merged_push_vals;
  my_cache.inner_barrier_.Resume();
  // barrier wait set buffer ptr
  barrier_.wait();
  my_cache.inner_barrier_.Pause();
  gather_inner_data_p2p(push_size,
                        d_keys,
                        d_push_vals,
                        res,
                        gpu_id,
                        device_num_,
                        trans_id,
                        value_bytes,
                        stream);
  // barrier wait all gpu aync memcpy data
  my_cache.inner_barrier_.Resume();
  barrier_.wait();
  my_cache.inner_barrier_.Pause();
  // all embedx merge
  size_t total_push_size = merge_grad(gpu_id,
                                      shard_recv_offset,
                                      my_cache.d_merged_keys,
                                      my_cache.d_merged_push_keys,
                                      my_cache.d_merged_vals,
                                      my_cache.d_merged_push_vals,
                                      stream);
  return total_push_size;
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    gather_sparse_gradient_by_all2all(const int &gpu_id,
                                      const size_t &fea_size,
                                      const KeyType *d_keys,
                                      const char *d_push_vals,
                                      const size_t &value_bytes,
                                      KeyType *d_out_keys,
                                      KeyType *d_tmp_keys,
                                      char *d_out_vals,
                                      char *d_tmp_vals,
                                      const cudaStream_t &stream) {
  auto &my_cache = storage_[gpu_id];
  my_cache.init_shard(fea_size, node_size_);
  auto &res = my_cache.shard_res;

  size_t *h_local_part_sizes = res.h_local_part_sizes.data();
  size_t *h_local_part_offsets = res.h_local_part_offsets.data();
  uint32_t *h_push_fea_sizes = res.h_push_fea_sizes.data();

  partition_shard_keys(gpu_id,
                       fea_size,
                       d_keys,
                       res.d_local_idx_parted,
                       d_tmp_keys,
                       h_local_part_sizes,
                       node_size_,
                       stream);
  int all_shard_part_size = node_size_ * node_size_;
  h_local_part_offsets[0] = 0;
  for (int i = 0; i < node_size_; i++) {
    int offset = rank_id_ * node_size_ + i;
    h_push_fea_sizes[offset] = h_local_part_sizes[i];
    h_local_part_offsets[i + 1] =
        h_local_part_offsets[i] + h_local_part_sizes[i];
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(res.d_node_size_ptr,
                                             &h_push_fea_sizes[0],
                                             all_shard_part_size * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  my_cache.node_barrier_.Resume();
  auto &comm = nccl_inter_comms_[gpu_id];
  auto nccl_stream = resource_->comm_stream(gpu_id, 0);
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclAllGather(&res.d_node_size_ptr[rank_id_ * node_size_],
                                  reinterpret_cast<void *>(res.d_node_size_ptr),
                                  node_size_,
                                  ncclInt,
                                  comm,
                                  nccl_stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(nccl_stream));
  my_cache.node_barrier_.Pause();

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&h_push_fea_sizes[0],
                                             res.d_node_size_ptr,
                                             all_shard_part_size * sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  size_t *h_remote_part_sizes = res.h_remote_part_sizes.data();
  size_t *h_remote_part_offsets = res.h_remote_part_offsets.data();

  h_remote_part_offsets[0] = 0;
  for (int i = 0; i < node_size_; i++) {
    int offset = node_size_ * i + rank_id_;
    int recv_num = h_push_fea_sizes[offset];
    h_remote_part_sizes[i] = recv_num;
    h_remote_part_offsets[i + 1] = h_remote_part_offsets[i] + recv_num;
  }
  size_t total_recv_fea_num = h_remote_part_offsets[node_size_];
  // my_cache.alloc(total_recv_fea_num, max_type_size_,
  // HeterCommType::COPY_ALL);
  my_cache.check(total_recv_fea_num, max_type_size_);
  // fill shard vals
  heter_comm_kernel_->gather_vals(
      reinterpret_cast<float *>(d_tmp_vals),         // out
      reinterpret_cast<const float *>(d_push_vals),  // in
      res.d_local_idx_parted,
      fea_size,
      value_bytes,
      stream);

  size_t total_send_recv = 0;
  if (rdma_checker_->need_rdma_trans()) {
    total_send_recv = send_gradient_by_all2all_trans(gpu_id,
                                                     rank_id_,
                                                     node_size_,
                                                     fea_size,
                                                     d_tmp_keys,
                                                     d_tmp_vals,
                                                     value_bytes,
                                                     d_out_keys,
                                                     d_out_vals,
                                                     stream);
  } else {
    // send local device
    my_cache.node_trans_.Resume();
    total_send_recv =
        send_data_by_all2all(gpu_id,
                             node_size_,
                             rank_id_,
                             sizeof(KeyType),
                             h_local_part_sizes,
                             h_local_part_offsets,
                             h_remote_part_sizes,
                             h_remote_part_offsets,
                             reinterpret_cast<const char *>(d_tmp_keys),
                             reinterpret_cast<char *>(d_out_keys),
                             stream);
    send_data_by_all2all(gpu_id,
                         node_size_,
                         rank_id_,
                         value_bytes,
                         h_local_part_sizes,
                         h_local_part_offsets,
                         h_remote_part_sizes,
                         h_remote_part_offsets,
                         reinterpret_cast<const char *>(d_tmp_vals),
                         reinterpret_cast<char *>(d_out_vals),
                         stream);
    my_cache.node_trans_.Pause();
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  return total_recv_fea_num;
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t
HeterComm<KeyType, ValType, GradType, GPUAccessor>::send_keys_by_all2all_trans(
    const int &gpu_id,
    const int &nccl_rank_id,
    const int &nccl_node_size,
    const size_t &fea_size,
    const KeyType *d_in_keys,
    KeyType *d_out_keys,
    const cudaStream_t &stream) {
  size_t total_fea_num = 0;
  auto &my_cache = storage_[gpu_id];
  if (!rdma_checker_->is_device_support_rdma(gpu_id)) {
    // AnyDeviceGuard guard(resource_->dev_id(gpu_id));
    int trans_id = get_transfer_devid(gpu_id);
    auto &trans = storage_[trans_id];
    // wait node alloc hbm
    trans.sem_wait->post();
    my_cache.sem_wait->wait();

    const size_t &recv_size =
        my_cache.shard_res.h_remote_part_offsets[nccl_node_size];
    size_t need_len = std::max(fea_size, recv_size);
    PADDLE_ENFORCE_EQ(
        trans.trans_keys_buff->size() >= need_len * sizeof(KeyType) * 2,
        true,
        common::errors::InvalidArgument(
            "The size of trnas keys buffer should be greater than or equal to "
            "%d, but got %d.",
            need_len * sizeof(KeyType) * 2,
            trans.trans_keys_buff->size()));

    // p2p copy
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(trans.d_merged_trans_keys,
                                                   trans_id,
                                                   d_in_keys,
                                                   gpu_id,
                                                   fea_size * sizeof(KeyType),
                                                   stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

    // wait node data ok
    trans.sem_wait->post();
    my_cache.sem_wait->wait();

    // p2p copy
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyPeerAsync(d_out_keys,
                            gpu_id,
                            trans.d_merged_push_trans_keys,
                            trans_id,
                            recv_size * sizeof(KeyType),
                            stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  } else {
    my_cache.sem_wait->wait();
    int trans_id = get_transfer_devid(gpu_id);
    auto &trans = storage_[trans_id];

    // alloc trans mem
    size_t trans_len =
        std::max(trans.shard_res.h_local_part_offsets[nccl_node_size],
                 trans.shard_res.h_remote_part_offsets[nccl_node_size]);
    my_cache.init_trans(trans_len, max_type_size_);

    trans.sem_wait->post();
    my_cache.sem_wait->wait();

    // send local device
    my_cache.node_trans_.Resume();
    total_fea_num =
        send_data_by_all2all(gpu_id,
                             nccl_node_size,
                             nccl_rank_id,
                             sizeof(KeyType),
                             my_cache.shard_res.h_local_part_sizes.data(),
                             my_cache.shard_res.h_local_part_offsets.data(),
                             my_cache.shard_res.h_remote_part_sizes.data(),
                             my_cache.shard_res.h_remote_part_offsets.data(),
                             reinterpret_cast<const char *>(d_in_keys),
                             reinterpret_cast<char *>(d_out_keys),
                             stream);
    // send trans device
    total_fea_num += send_data_by_all2all(
        gpu_id,
        nccl_node_size,
        nccl_rank_id,
        sizeof(KeyType),
        trans.shard_res.h_local_part_sizes.data(),
        trans.shard_res.h_local_part_offsets.data(),
        trans.shard_res.h_remote_part_sizes.data(),
        trans.shard_res.h_remote_part_offsets.data(),
        reinterpret_cast<const char *>(my_cache.d_merged_trans_keys),
        reinterpret_cast<char *>(my_cache.d_merged_push_trans_keys),
        stream);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    my_cache.node_trans_.Pause();

    trans.sem_wait->post();
  }
  return total_fea_num;
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t
HeterComm<KeyType, ValType, GradType, GPUAccessor>::send_vals_by_all2all_trans(
    const int &gpu_id,
    const int &nccl_rank_id,
    const int &nccl_node_size,
    const char *d_in_vals,
    char *d_out_vals,
    const size_t &value_bytes,
    const cudaStream_t &stream) {
  auto &my_cache = storage_[gpu_id];
  auto h_local_part_sizes = my_cache.shard_res.h_local_part_sizes.data();
  auto h_local_part_offsets = my_cache.shard_res.h_local_part_offsets.data();
  auto h_remote_part_sizes = my_cache.shard_res.h_remote_part_sizes.data();
  auto h_remote_part_offsets = my_cache.shard_res.h_remote_part_offsets.data();

  size_t total_fea_num = 0;
  if (!rdma_checker_->is_device_support_rdma(gpu_id)) {
    // AnyDeviceGuard guard(resource_->dev_id(gpu_id));
    int trans_id = get_transfer_devid(gpu_id);
    auto &trans = storage_[trans_id];

    const size_t &send_size = h_remote_part_offsets[nccl_node_size];
    // p2p copy
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(trans.d_merged_trans_vals,
                                                   trans_id,
                                                   d_in_vals,
                                                   gpu_id,
                                                   send_size * value_bytes,
                                                   stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

    // wait node data ok
    trans.sem_wait->post();
    my_cache.sem_wait->wait();

    const size_t &recv_size = h_local_part_offsets[nccl_node_size];
    // p2p copy
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyPeerAsync(d_out_vals,
                            gpu_id,
                            trans.d_merged_push_trans_vals,
                            trans_id,
                            recv_size * value_bytes,
                            stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  } else {
    my_cache.sem_wait->wait();
    int trans_id = get_transfer_devid(gpu_id);
    auto &trans = storage_[trans_id];

    // send local device
    my_cache.node_trans_.Resume();
    total_fea_num =
        send_data_by_all2all(gpu_id,
                             nccl_node_size,
                             nccl_rank_id,
                             value_bytes,
                             h_remote_part_sizes,
                             h_remote_part_offsets,
                             h_local_part_sizes,
                             h_local_part_offsets,
                             reinterpret_cast<const char *>(d_in_vals),
                             reinterpret_cast<char *>(d_out_vals),
                             stream);
    // send trans device
    total_fea_num += send_data_by_all2all(
        gpu_id,
        nccl_node_size,
        nccl_rank_id,
        value_bytes,
        trans.shard_res.h_remote_part_sizes.data(),
        trans.shard_res.h_remote_part_offsets.data(),
        trans.shard_res.h_local_part_sizes.data(),
        trans.shard_res.h_local_part_offsets.data(),
        reinterpret_cast<const char *>(my_cache.d_merged_trans_vals),
        reinterpret_cast<char *>(my_cache.d_merged_push_trans_vals),
        stream);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    my_cache.node_trans_.Pause();
    trans.sem_wait->post();
  }
  return total_fea_num;
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    send_gradient_by_all2all_trans(const int &gpu_id,
                                   const int &nccl_rank_id,
                                   const int &nccl_node_size,
                                   const size_t &fea_size,
                                   const KeyType *d_in_keys,
                                   const char *d_in_vals,
                                   const size_t &value_bytes,
                                   KeyType *d_out_keys,
                                   char *d_out_vals,
                                   const cudaStream_t &stream) {
  auto &my_cache = storage_[gpu_id];
  size_t total_send_recv = 0;
  if (!rdma_checker_->is_device_support_rdma(gpu_id)) {
    // AnyDeviceGuard guard(resource_->dev_id(gpu_id));
    int trans_id = get_transfer_devid(gpu_id);
    auto &trans = storage_[trans_id];

    // wait node alloc hbm
    // trans.sem_wait->post();
    // my_cache.sem_wait->wait();
    const size_t &recv_total_size =
        my_cache.shard_res.h_remote_part_offsets[nccl_node_size];
    size_t need_len = std::max(fea_size, recv_total_size);
    PADDLE_ENFORCE_EQ(
        trans.trans_keys_buff->size() >= need_len * sizeof(KeyType) * 2,
        true,
        common::errors::InvalidArgument(
            "The size of trans keys buffer should be greater than or equal to "
            "%d, but got %d.",
            need_len * sizeof(KeyType) * 2,
            trans.trans_keys_buff->size()));

    // p2p copy
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(trans.d_merged_trans_keys,
                                                   trans_id,
                                                   d_in_keys,
                                                   gpu_id,
                                                   fea_size * sizeof(KeyType),
                                                   stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(trans.d_merged_trans_vals,
                                                   trans_id,
                                                   d_in_vals,
                                                   gpu_id,
                                                   fea_size * value_bytes,
                                                   stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

    // wait node data ok
    trans.sem_wait->post();
    my_cache.sem_wait->wait();

    // p2p copy
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyPeerAsync(d_out_keys,
                            gpu_id,
                            trans.d_merged_push_trans_keys,
                            trans_id,
                            recv_total_size * sizeof(KeyType),
                            stream));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyPeerAsync(d_out_vals,
                            gpu_id,
                            trans.d_merged_push_trans_vals,
                            trans_id,
                            recv_total_size * value_bytes,
                            stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  } else {
    // copy local rank id data
    my_cache.sem_wait->wait();
    int trans_id = get_transfer_devid(gpu_id);
    auto &trans = storage_[trans_id];

    //    size_t trans_len =
    //    std::max(trans.shard_res.h_local_part_offsets[nccl_node_size],
    //            trans.shard_res.h_remote_part_offsets[nccl_node_size]);
    //    // alloc mem
    //    my_cache.init_trans(trans_len, value_bytes);
    //
    //    trans.sem_wait->post();
    //    my_cache.sem_wait->wait();

    my_cache.node_trans_.Resume();
    // send local device
    total_send_recv =
        send_data_by_all2all(gpu_id,
                             nccl_node_size,
                             nccl_rank_id,
                             sizeof(KeyType),
                             my_cache.shard_res.h_local_part_sizes.data(),
                             my_cache.shard_res.h_local_part_offsets.data(),
                             my_cache.shard_res.h_remote_part_sizes.data(),
                             my_cache.shard_res.h_remote_part_offsets.data(),
                             reinterpret_cast<const char *>(d_in_keys),
                             reinterpret_cast<char *>(d_out_keys),
                             stream);
    send_data_by_all2all(gpu_id,
                         nccl_node_size,
                         nccl_rank_id,
                         value_bytes,
                         my_cache.shard_res.h_local_part_sizes.data(),
                         my_cache.shard_res.h_local_part_offsets.data(),
                         my_cache.shard_res.h_remote_part_sizes.data(),
                         my_cache.shard_res.h_remote_part_offsets.data(),
                         reinterpret_cast<const char *>(d_in_vals),
                         reinterpret_cast<char *>(d_out_vals),
                         stream);
    // send trans device
    total_send_recv += send_data_by_all2all(
        gpu_id,
        nccl_node_size,
        nccl_rank_id,
        sizeof(KeyType),
        trans.shard_res.h_local_part_sizes.data(),
        trans.shard_res.h_local_part_offsets.data(),
        trans.shard_res.h_remote_part_sizes.data(),
        trans.shard_res.h_remote_part_offsets.data(),
        reinterpret_cast<const char *>(my_cache.d_merged_trans_keys),
        reinterpret_cast<char *>(my_cache.d_merged_push_trans_keys),
        stream);
    send_data_by_all2all(
        gpu_id,
        nccl_node_size,
        nccl_rank_id,
        value_bytes,
        trans.shard_res.h_local_part_sizes.data(),
        trans.shard_res.h_local_part_offsets.data(),
        trans.shard_res.h_remote_part_sizes.data(),
        trans.shard_res.h_remote_part_offsets.data(),
        reinterpret_cast<const char *>(my_cache.d_merged_trans_vals),
        reinterpret_cast<char *>(my_cache.d_merged_push_trans_vals),
        stream);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    my_cache.node_trans_.Pause();
    trans.sem_wait->post();
  }
  return total_send_recv;
}
}  // namespace framework
}  // namespace paddle
#endif
