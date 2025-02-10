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

#include "paddle/phi/core/distributed/comm_context_manager.h"

#include <memory>
#include <string>
#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_GLOO)
#include <gloo/rendezvous/prefix_store.h>
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#include "paddle/phi/core/distributed/gloo_utils.h"
#include "paddle/phi/core/distributed/store/gloo_store.h"
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/distributed/nccl_tools.h"
#elif defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#endif
#ifdef PADDLE_WITH_UCC
#include "paddle/phi/core/distributed/ucc_comm_context.h"
#endif

namespace phi::distributed {

int CommContextManager::device_id = -1;

void CommContextManager::SetDeviceId(int dev_id) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  phi::backends::gpu::SetDeviceId(dev_id);
  CommContextManager::device_id = dev_id;
#elif defined(PADDLE_WITH_XPU_BKCL)
  phi::backends::xpu::SetXPUDeviceId(dev_id);
  CommContextManager::device_id = dev_id;
#endif
}

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
void CommContextManager::CreateNCCLCommContext(
    const std::shared_ptr<Store>& store,
    const std::string& unique_comm_key,
    int rank,
    int size,
    const std::string& hash_key,
    const P2POption* p2p_opt,
    int nccl_comm_init_option) {
  auto& comm_context_manager = CommContextManager::GetInstance();
  if (comm_context_manager.Has(unique_comm_key)) {
    return;
  }
  ncclUniqueId nccl_id;
  if (rank == 0 || (p2p_opt && p2p_opt->is_p2p_op && p2p_opt->p2p_rank == 0)) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGetUniqueId(&nccl_id));
  }

  std::string unique_key = "NCCLCommContext/" + unique_comm_key + hash_key;
  if (rank == 0 || (p2p_opt && p2p_opt->is_p2p_op && p2p_opt->p2p_rank == 0)) {
    std::vector<uint8_t> nccl_id_wrapper(
        reinterpret_cast<uint8_t*>(&nccl_id),
        reinterpret_cast<uint8_t*>(&nccl_id) + NCCL_UNIQUE_ID_BYTES);
    store->set(unique_key, nccl_id_wrapper);
  } else {
    const auto& nccl_id_wrapper = store->get(unique_key);
    std::memcpy(&nccl_id, nccl_id_wrapper.data(), nccl_id_wrapper.size());
  }

  if (p2p_opt) {
    rank = p2p_opt->rank;
    size = p2p_opt->num_ranks;
  }
  VLOG(3) << "init NCCLCommContext rank: " << rank << ", size: " << size
          << ", unique_comm_key: " << unique_comm_key
          << ", unique_key: " << unique_key
          << ", nccl_id: " << SerializeNCCLUniqueId(nccl_id);
  auto nccl_comm_context = std::make_unique<NCCLCommContext>(
      rank, size, nccl_id, nccl_comm_init_option);
  if (CommContextManager::device_id != -1) {
    std::unique_ptr<phi::GPUContext> dev_ctx(
        new phi::GPUContext(phi::GPUPlace(CommContextManager::device_id)));
    dev_ctx->SetAllocator(phi::memory_utils::GetAllocator(
        CommContextManager::device_id, dev_ctx->stream()));
    dev_ctx->SetHostAllocator(phi::memory_utils::GetHostAllocator());
    dev_ctx->SetZeroAllocator(
        phi::memory_utils::GetZeroAllocator(CommContextManager::device_id));
    dev_ctx->SetHostZeroAllocator(phi::memory_utils::GetHostZeroAllocator());
    dev_ctx->SetPinnedAllocator(phi::memory_utils::GetPinnedAllocator());
    dev_ctx->PartialInitWithAllocator();
    auto compute_event =
        phi::memory_utils::GetCudaEvent(CommContextManager::device_id);
    auto comm_event =
        phi::memory_utils::GetCudaEvent(CommContextManager::device_id);

    nccl_comm_context->SetDevContext(std::move(dev_ctx));
    nccl_comm_context->SetComputeEvent(std::move(compute_event));
    nccl_comm_context->SetCommEvent(std::move(comm_event));
  }

  comm_context_manager.SetStore(store);
  comm_context_manager.Emplace(unique_comm_key, std::move(nccl_comm_context));
}
#endif

#if defined(PADDLE_WITH_GLOO)
void CommContextManager::CreateGlooCommContext(
    const std::shared_ptr<Store>& store,
    const std::string& unique_comm_key,
    int rank,
    int size) {
  GlooStore store_wrapper(store);
  auto gloo_store = std::make_shared<gloo::rendezvous::PrefixStore>(
      unique_comm_key, store_wrapper);

  auto gloo_device = CreateGlooDevice();

  auto gloo_comm_context =
      std::make_unique<GlooCommContext>(rank, size, gloo_store, gloo_device);
  auto& comm_context_manager = CommContextManager::GetInstance();
  // set actual store to manager
  comm_context_manager.SetStore(store);
  comm_context_manager.Emplace(unique_comm_key, std::move(gloo_comm_context));
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
void CommContextManager::CreateXCCLCommContext(
    const std::shared_ptr<Store>& store,
    const std::string& unique_comm_key,
    const phi::Place& place,
    int rank,
    int size,
    const std::string& hash_key) {
  phi::ccl::CCLRootId xccl_root_id;
  if (rank == 0) {
    phi::DeviceManager::CCLGetUniqueId(place.GetDeviceType(), &xccl_root_id);
  }

  std::string unique_key = "XCCLCommContext/" + unique_comm_key;
  if (!hash_key.empty()) {
    unique_key += "/" + hash_key;
  }
  if (rank == 0) {
    store->set(unique_key, xccl_root_id);
  } else {
    xccl_root_id = store->get(unique_key);
  }
  VLOG(3) << "init xccl rank: " << rank << ", nranks: " << size
          << ", unique_comm_key: " << unique_comm_key << ", xccl uniqueid: "
          << phi::ccl::SerializeXCCLUniqueId(xccl_root_id);
  auto xccl_comm_context =
      std::make_unique<XCCLCommContext>(place, rank, size, xccl_root_id);
  auto& comm_context_manager = CommContextManager::GetInstance();
  comm_context_manager.SetStore(store);
  comm_context_manager.Emplace(unique_comm_key, std::move(xccl_comm_context));
}
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
void CommContextManager::CreateBKCLCommContext(
    const std::shared_ptr<Store>& store,
    const std::string& unique_comm_key,
    int rank,
    int size,
    const std::string& hash_key) {
  auto& comm_context_manager = CommContextManager::GetInstance();
  if (comm_context_manager.Has(unique_comm_key)) {
    return;
  }
  BKCLUniqueId bkcl_id;
  if (rank == 0) {
    PADDLE_ENFORCE_XPU_SUCCESS(bkcl_get_unique_id(&bkcl_id));
  }

  std::string unique_key = "BKCLCommContext/" + unique_comm_key + hash_key;
  if (rank == 0) {
    std::vector<uint8_t> bkcl_id_wrapper(
        reinterpret_cast<uint8_t*>(&bkcl_id),
        reinterpret_cast<uint8_t*>(&bkcl_id) + BKCL_UNIQUE_ID_BYTES);
    store->set(unique_key, bkcl_id_wrapper);
  } else {
    const auto& bkcl_id_wrapper = store->get(unique_key);
    std::memcpy(&bkcl_id, bkcl_id_wrapper.data(), bkcl_id_wrapper.size());
  }

  VLOG(3) << "init BKCLCommContext rank: " << rank << ", size: " << size
          << ", unique_comm_key: " << unique_comm_key
          << ", unique_key: " << unique_key;
  auto bkcl_comm_context =
      std::make_unique<BKCLCommContext>(rank, size, bkcl_id);

  if (CommContextManager::device_id != -1) {
    std::unique_ptr<phi::XPUContext> dev_ctx(new phi::XPUContext(
        phi::XPUPlace(CommContextManager::device_id), true));
    dev_ctx->SetAllocator(phi::memory_utils::GetAllocator(
        CommContextManager::device_id, dev_ctx->stream()));
    dev_ctx->SetHostAllocator(phi::memory_utils::GetHostAllocator());
    dev_ctx->SetZeroAllocator(
        phi::memory_utils::GetZeroAllocator(CommContextManager::device_id));
    dev_ctx->SetHostZeroAllocator(phi::memory_utils::GetHostZeroAllocator());
    // XPUs do not have the concept of pinned memory,
    // so the get_pinned_allocator function is not set.

    // It currently does not support dev_ctx->PartialInitWithAllocator().
    auto compute_event =
        phi::memory_utils::GetXpuEvent(CommContextManager::device_id);
    auto comm_event =
        phi::memory_utils::GetXpuEvent(CommContextManager::device_id);

    bkcl_comm_context->SetDevContext(std::move(dev_ctx));
    bkcl_comm_context->SetComputeEvent(std::move(compute_event));
    bkcl_comm_context->SetCommEvent(std::move(comm_event));
  }

  comm_context_manager.SetStore(store);
  comm_context_manager.Emplace(unique_comm_key, std::move(bkcl_comm_context));
}
#endif

#if defined(PADDLE_WITH_UCC)
void CommContextManager::CreateUCCCommContext(
    const std::shared_ptr<Store>& store,
    const std::string& unique_comm_key,
    int rank,
    int size,
    const std::string& hash_key) {
  auto& comm_context_manager = CommContextManager::GetInstance();
  if (comm_context_manager.Has(unique_comm_key)) {
    return;
  }

  VLOG(3) << "init UCCCommContext rank: " << rank << ", size: " << size;
  auto ucc_comm_context = std::make_unique<UCCCommContext>(store, rank, size);

  // if (CommContextManager::device_id != -1) {
  //   std::unique_ptr<phi::GPUContext> dev_ctx(
  //       new phi::GPUContext(phi::GPUPlace(CommContextManager::device_id)));
  //   dev_ctx->SetAllocator(phi::memory_utils::GetAllocator(
  //       CommContextManager::device_id, dev_ctx->stream()));
  //   dev_ctx->SetHostAllocator(phi::memory_utils::GetHostAllocator());
  //   dev_ctx->SetZeroAllocator(
  //       phi::memory_utils::GetZeroAllocator(CommContextManager::device_id));
  //   dev_ctx->SetHostZeroAllocator(phi::memory_utils::GetHostZeroAllocator());
  //   dev_ctx->SetPinnedAllocator(phi::memory_utils::GetPinnedAllocator());
  //   dev_ctx->PartialInitWithAllocator();
  //   auto compute_event =
  //       phi::memory_utils::GetCudaEvent(CommContextManager::device_id);
  //   auto comm_event =
  //       phi::memory_utils::GetCudaEvent(CommContextManager::device_id);

  //   nccl_comm_context->SetDevContext(std::move(dev_ctx));
  //   nccl_comm_context->SetComputeEvent(std::move(compute_event));
  //   nccl_comm_context->SetCommEvent(std::move(comm_event));
  // }

  comm_context_manager.SetStore(store);
  comm_context_manager.Emplace(unique_comm_key, std::move(ucc_comm_context));
}
#endif

CommContext* CommContextManager::Emplace(
    const std::string& unique_comm_key,
    std::unique_ptr<CommContext> comm_context) {
  PADDLE_ENFORCE_EQ(
      id_to_comm_context_.find(unique_comm_key),
      id_to_comm_context_.end(),
      errors::AlreadyExists("The unique key %s already exists in the map.",
                            unique_comm_key));
  id_to_comm_context_.emplace(unique_comm_key, std::move(comm_context));
  return id_to_comm_context_.at(unique_comm_key).get();
}

CommContext* CommContextManager::Get(const std::string& unique_comm_key) const {
  PADDLE_ENFORCE_NE(
      id_to_comm_context_.find(unique_comm_key),
      id_to_comm_context_.end(),
      errors::NotFound("Can not find unique key %s in map.", unique_comm_key));

  return id_to_comm_context_.at(unique_comm_key).get();
}

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
int CommContextManager::GetRingId(const ncclComm_t& comm) const {
  for (const auto& iter : id_to_comm_context_) {
    if (static_cast<phi::distributed::NCCLCommContext*>(iter.second.get())
            ->GetNcclComm() == comm) {
      return std::stoi(iter.first);
    }
  }
  return -1;
}
#endif

bool CommContextManager::Has(const std::string& unique_comm_key) const {
  return id_to_comm_context_.find(unique_comm_key) != id_to_comm_context_.end();
}

void CommContextManager::SetGroupSize(const std::string& pg_key, int size) {
  pg_key_size_[pg_key] = size;
}

void CommContextManager::AddGroupRanks(const std::string& pg_key,
                                       std::vector<int> global_ranks) {
  if (pg_key_ranks_.find(pg_key) == pg_key_ranks_.end()) {
    pg_key_ranks_[pg_key] = global_ranks;
  }
}

std::vector<int> CommContextManager::GetGroupRanks(
    const std::string& pg_key) const {
  PADDLE_ENFORCE_NE(
      pg_key_ranks_.find(pg_key),
      pg_key_ranks_.end(),
      errors::NotFound("Can not find pg_key %d in GroupRanks.", pg_key));
  return pg_key_ranks_.at(pg_key);
}

}  // namespace phi::distributed
