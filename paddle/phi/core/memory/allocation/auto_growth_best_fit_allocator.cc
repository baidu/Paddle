// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/auto_growth_best_fit_allocator.h"

#include <algorithm>
#include <mutex>  // NOLINT
#include <utility>

#include "paddle/common/flags.h"
#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/memory/allocation/aligned_allocator.h"
#include "paddle/phi/core/memory/stats.h"
PHI_DEFINE_EXPORTED_READONLY_bool(
    free_idle_chunk,
    false,
    "Whether to free idle chunk when each allocation is freed. "
    "If false, all freed allocation would be cached to speed up next "
    "allocation request. If true, no allocation would be cached. This "
    "flag only works when FLAGS_allocator_strategy=auto_growth.");

PHI_DEFINE_EXPORTED_READONLY_bool(
    free_when_no_cache_hit,
    false,
    "Whether to free idle chunks when no cache hit. If true, idle "
    "chunk would be freed when no cache hit; if false, idle "
    "chunk would be freed when out of memory occurs. This flag "
    "only works when FLAGS_allocator_strategy=auto_growth.");

PHI_DEFINE_EXPORTED_READONLY_bool(print_allocator_trace_info,
                                  false,
                                  "print trace memory info");

PHI_DEFINE_EXPORTED_READONLY_bool(dump_chunk_info, false, "dump chunk info");
namespace paddle::memory::allocation {

AutoGrowthBestFitAllocator::AutoGrowthBestFitAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    size_t alignment,
    size_t chunk_size,
    bool allow_free_idle_chunk,
    int extra_padding_size)
    : underlying_allocator_(std::move(underlying_allocator)),
      alignment_(alignment),
      chunk_size_(std::max(AlignedSize(chunk_size, alignment), alignment)),
      allow_free_idle_chunk_(allow_free_idle_chunk),
      extra_padding_size_(extra_padding_size) {
  total_alloc_times_ = 0;
  total_alloc_size_ = 0;
  total_free_times_ = 0;
  total_free_size_ = 0;
  VLOG(4) << "chunk_size_:" << chunk_size_;
}

void AutoGrowthBestFitAllocator::DumpInfo() const {
  for (auto chunk_it = chunks_.begin(); chunk_it != chunks_.end(); ++chunk_it) {
    std::cout << "Chunk\t";
    std::ostringstream oss_used;
    std::ostringstream oss_free;
    size_t total = 0, free = 0, used = 0;
    for (auto &b : chunk_it->blocks_) {
      total += b.size_;
      if (b.is_free_) {
        free += b.size_;
        oss_free << "(" << b.size_ << "," << b.ptr_ << ")";
      } else {
        used += b.size_;
        oss_used << "(" << b.size_ << "," << b.ptr_ << ")";
      }
    }
    std::cout << total << "\t" << used << "\t" << free << "\t";
    std::cout << "[" << oss_used.str() << "]\t[" << oss_free.str() << "]"
              << std::endl;
  }
}
phi::Allocation *AutoGrowthBestFitAllocator::AllocateImpl(
    size_t unaligned_size) {
  phi::RecordEvent record("AutoGrowthBestFitAllocator::Allocate",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);

  size_t size = AlignedSize(unaligned_size + extra_padding_size_, alignment_);

  VLOG(10) << "Allocate " << unaligned_size << " bytes, aligned to " << size
           << ", extra size " << extra_padding_size_;

  std::lock_guard<SpinLock> guard(spinlock_);
  auto iter = free_blocks_.lower_bound(std::make_pair(size, nullptr));
  BlockIt block_it;
  if (iter != free_blocks_.end()) {
    block_it = iter->second;
    free_blocks_.erase(iter);
    auto *chunk = block_it->chunk_;
    size_t remaining_size = block_it->size_ - size;
    VLOG(10) << "Allocate " << size << " bytes from chunk size "
             << block_it->size_ << ", remaining " << remaining_size;
    if (remaining_size == 0) {
      block_it->is_free_ = false;
    } else {
      auto remaining_free_block = chunk->blocks_.insert(
          block_it, Block(block_it->ptr_, remaining_size, true, chunk));
      free_blocks_.emplace(std::make_pair(remaining_size, block_it->ptr_),
                           remaining_free_block);
      block_it->ptr_ =
          reinterpret_cast<uint8_t *>(block_it->ptr_) + remaining_size;
      block_it->size_ = size;
      block_it->is_free_ = false;
    }
  } else {
    if (FLAGS_dump_chunk_info) {
      std::cout << "MemDbg memory not enough growth chunk, need size = " << size
                << std::endl;
      DumpInfo();
    }

    if (FLAGS_free_when_no_cache_hit) {
      FreeIdleChunks();
    }
    size_t realloc_size = std::max(size, chunk_size_);

    try {
      chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
          underlying_allocator_->Allocate(realloc_size)));
    } catch (BadAlloc &ex) {
      if (FLAGS_dump_chunk_info) {
        std::cout << "MemDbg OOM" << std::endl;
        DumpInfo();
      }
      if (FLAGS_free_when_no_cache_hit) throw ex;
      FreeIdleChunks();
      chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
          underlying_allocator_->Allocate(realloc_size)));
    }

    auto *chunk = &(*chunks_.rbegin());
    realloc_size = chunk->allocation_->size();
    uint8_t *p = reinterpret_cast<uint8_t *>(chunk->allocation_->ptr());
    auto &blocks = chunk->blocks_;

    size_t remaining_size = realloc_size - size;
    if (remaining_size > 0) {
      blocks.emplace_back(p, remaining_size, true, chunk);
      free_blocks_.emplace(std::make_pair(remaining_size, p), --(blocks.end()));
    }
    blocks.emplace_back(p + remaining_size, size, false, chunk);
    block_it = --(blocks.end());
    VLOG(2) << "Not found and reallocate " << realloc_size << "("
            << static_cast<void *>(p) << "), and remaining " << remaining_size;
    if (FLAGS_dump_chunk_info) {
      std::cout << "MemDbg memory after growth chunk, realloc_size = "
                << realloc_size << std::endl;
      DumpInfo();
    }
  }
  ++total_alloc_times_;
  total_alloc_size_ += size;
  VLOG(10) << "Alloc " << block_it->size_ << " bytes, ptr = " << block_it->ptr_;
  return new BlockAllocation(block_it);
}

void AutoGrowthBestFitAllocator::FreeImpl(phi::Allocation *allocation) {
  phi::RecordEvent record("AutoGrowthBestFitAllocator::Free",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);
  VLOG(10) << "Free " << allocation->size()
           << " bytes, ptr = " << allocation->ptr();
  std::lock_guard<SpinLock> guard(spinlock_);
  auto block_it = static_cast<BlockAllocation *>(allocation)->block_it_;
  auto &blocks = block_it->chunk_->blocks_;

  total_free_times_ += 1;
  total_free_size_ += block_it->size_;

  block_it->is_free_ = true;

  if (block_it != blocks.begin()) {
    auto prev_it = block_it;
    --prev_it;

    if (prev_it->is_free_) {
      free_blocks_.erase(std::make_pair(prev_it->size_, prev_it->ptr_));
      prev_it->size_ += block_it->size_;
      blocks.erase(block_it);
      block_it = prev_it;
    }
  }

  auto next_it = block_it;
  ++next_it;

  if (next_it != blocks.end() && next_it->is_free_) {
    free_blocks_.erase(std::make_pair(next_it->size_, next_it->ptr_));
    block_it->size_ += next_it->size_;
    blocks.erase(next_it);
  }

  free_blocks_.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                       block_it);
  if (phi::is_cpu_place(allocation->place()) ||
      phi::is_cuda_pinned_place(allocation->place())) {
    HOST_MEMORY_STAT_UPDATE(
        Allocated, allocation->place().GetDeviceId(), -allocation->size());
  } else {
    DEVICE_MEMORY_STAT_UPDATE(
        Allocated, allocation->place().GetDeviceId(), -allocation->size());
  }
  delete allocation;

  if (FLAGS_free_idle_chunk) {
    FreeIdleChunks();
  }
}

uint64_t AutoGrowthBestFitAllocator::FreeIdleChunks() {
  if (FLAGS_dump_chunk_info) {
    std::cout << "FreeIdleChunks called" << std::endl;
  }
  if (!allow_free_idle_chunk_) {
    return 0;
  }
  uint64_t bytes = 0;
  for (auto chunk_it = chunks_.begin(); chunk_it != chunks_.end();) {
    auto &blocks = chunk_it->blocks_;
    if (blocks.size() == 1 && blocks.begin()->is_free_) {
      auto &block = *blocks.begin();
      VLOG(2) << "Free chunk with size " << block.size_;
      if (FLAGS_dump_chunk_info) {
        std::cout << "FreeIdleChunks chunk is " << block.size_ << ", "
                  << block.ptr_ << std::endl;
      }
      bytes += block.size_;
      free_blocks_.erase(std::make_pair(block.size_, block.ptr_));
      chunk_it = chunks_.erase(chunk_it);
    } else {
      ++chunk_it;
    }
  }

  if (FLAGS_print_allocator_trace_info) {
    Trace();
  }
  return bytes;
}

void AutoGrowthBestFitAllocator::Trace() const {
  size_t cur_idle_bytes = 0;
  auto it = free_blocks_.begin();
  for (; it != free_blocks_.end(); ++it) {
    cur_idle_bytes += it->second->size_;
  }

  VLOG(1) << "alloc:"
          << total_alloc_size_ / static_cast<double>(1024 * 1024)  // NOLINT
          << "m free:"
          << total_free_size_ / static_cast<double>(1024 * 1024)  // NOLINT
          << "m busy:"
          << (total_alloc_size_ - total_free_size_) /  // NOLINT
                 static_cast<double>(1024 * 1024)
          << "m idle:"
          << cur_idle_bytes / static_cast<double>(1024 * 1024)  // NOLINT
          << "m alloc_times:" << total_alloc_times_
          << " free_times:" << total_free_times_
          << " free_blocks_num:" << free_blocks_.size()
          << " curr_chunks_num:" << chunks_.size();
}

}  // namespace paddle::memory::allocation
