/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "glog/logging.h"

DEFINE_bool(free_idle_memory, false,
            "If it is true, Paddle will try to free idle memory trunks during "
            "running time.");

namespace paddle {
namespace memory {
namespace detail {

BuddyAllocator::BuddyAllocator(
    std::unique_ptr<SystemAllocator> system_allocator, size_t min_chunk_size,
    size_t first_allocate_chunk_size, size_t reallocate_chunk_size)
    : min_chunk_size_(min_chunk_size),
      first_allocate_chunk_size_(first_allocate_chunk_size),
      reallocate_chunk_size_(reallocate_chunk_size),
      max_chunk_size_(first_allocate_chunk_size),
      cache_(system_allocator->UseGpu()),
      system_allocator_(std::move(system_allocator)) {}

BuddyAllocator::~BuddyAllocator() {
  VLOG(10) << "BuddyAllocator Disconstructor makes sure that all of these "
              "have actually been freed";
  while (!pool_.empty()) {
    auto block = static_cast<MemoryBlock*>(std::get<2>(*pool_.begin()));
    auto desc = cache_.load(block);
    VLOG(10) << "Free from block (" << block << ", " << desc.size << ")";

    system_allocator_->Free(block, desc.size, desc.index);
    cache_.invalidate(block);
    pool_.erase(pool_.begin());
  }
}

inline size_t align(size_t size, size_t alignment) {
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

void* BuddyAllocator::Alloc(size_t unaligned_size) {
  // adjust allocation alignment
  size_t size =
      align(unaligned_size + sizeof(MemoryBlock::Desc), min_chunk_size_);

  // acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  VLOG(10) << "Allocate " << unaligned_size << " bytes from chunk size "
           << size;

  // if the allocation is huge, send directly to the system allocator
  if (size > max_chunk_size_) {
    VLOG(10) << "Allocate from system allocator.";
    return SystemAlloc(size, false);
  }

  // query and allocate from the existing chunk
  auto it = FindExistChunk(size);

  // refill the pool if failure
  if (it == pool_.end()) {
    it = RefillPool();
    // if still failure, try to allocate from SystemAllocator
    if (it == pool_.end()) {
      return SystemAlloc(size, false);
    }
  } else {
    VLOG(10) << "Allocation from existing memory block " << std::get<2>(*it)
             << " at address "
             << reinterpret_cast<MemoryBlock*>(std::get<2>(*it))->data();
  }

  total_used_ += size;
  total_free_ -= size;

  // split the allocation and return data for use
  return reinterpret_cast<MemoryBlock*>(SplitToAlloc(it, size))->data();
}

void BuddyAllocator::Free(void* p) {
  // Point back to metadata
  auto block = static_cast<MemoryBlock*>(p)->metadata();

  // Acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  VLOG(10) << "Free from address " << block;

  if (block->type(cache_) == MemoryBlock::UNMANAGED_HUGE_CHUNK) {
    VLOG(10) << "Free directly from system allocator";
    system_allocator_->Free(block, block->total_size(cache_),
                            block->index(cache_));

    // Invalidate GPU allocation from cache
    cache_.invalidate(block);

    return;
  }

  block->mark_as_free(&cache_);

  total_used_ -= block->total_size(cache_);
  total_free_ += block->total_size(cache_);

  // Trying to merge the right buddy
  if (block->has_right_buddy(cache_)) {
    VLOG(10) << "Merging this block " << block << " with its right buddy "
             << block->right_buddy(cache_);

    auto right_buddy = block->right_buddy(cache_);

    if (right_buddy->type(cache_) == MemoryBlock::FREE_CHUNK) {
      // Take away right buddy from pool
      pool_.erase(IndexSizeAddress(right_buddy->index(cache_),
                                   right_buddy->total_size(cache_),
                                   right_buddy));

      // merge its right buddy to the block
      block->merge(&cache_, right_buddy);
    }
  }

  // Trying to merge the left buddy
  if (block->has_left_buddy(cache_)) {
    VLOG(10) << "Merging this block " << block << " with its left buddy "
             << block->left_buddy(cache_);

    auto left_buddy = block->left_buddy(cache_);

    if (left_buddy->type(cache_) == MemoryBlock::FREE_CHUNK) {
      // Take away right buddy from pool
      pool_.erase(IndexSizeAddress(left_buddy->index(cache_),
                                   left_buddy->total_size(cache_), left_buddy));

      // merge the block to its left buddy
      left_buddy->merge(&cache_, block);
      block = left_buddy;
    }
  }

  // Dumping this block into pool
  VLOG(10) << "Inserting free block (" << block << ", "
           << block->total_size(cache_) << ")";
  pool_.insert(
      IndexSizeAddress(block->index(cache_), block->total_size(cache_), block));

  if (FLAGS_free_idle_memory) {
    // Clean up if existing too much free memory
    // Prefer freeing fallback allocation first
    CleanIdleFallBackAlloc();

    // Free normal allocation
    CleanIdleNormalAlloc();
  }
}

size_t BuddyAllocator::Used() { return total_used_; }
size_t BuddyAllocator::GetMinChunkSize() { return min_chunk_size_; }
size_t BuddyAllocator::GetMaxChunkSize() {
  std::lock_guard<std::mutex> lock(mutex_);
  return max_chunk_size_;
}

void* BuddyAllocator::SystemAlloc(size_t size, bool is_managed) {
  size_t index = 0;
  void* p = system_allocator_->Alloc(&index, size);

  VLOG(10) << "Allocated " << p << " from system allocator.";

  if (p == nullptr) return nullptr;

  static_cast<MemoryBlock*>(p)->init(
      &cache_, is_managed ? MemoryBlock::MANAGED_HUGE_CHUNK
                          : MemoryBlock::UNMANAGED_HUGE_CHUNK,
      index, size, nullptr, nullptr);

  return static_cast<MemoryBlock*>(p)->data();
}

BuddyAllocator::PoolSet::iterator BuddyAllocator::RefillPool() {
  if (total_used_ + total_free_ > 0) {
    max_chunk_size_ = reallocate_chunk_size_;
  }

  // Allocate a new maximum sized block
  size_t index = 0;
  size_t chunk_size = max_chunk_size_;
  void* p = system_allocator_->Alloc(&index, chunk_size);

  if (p == nullptr) return pool_.end();

  VLOG(10) << "Creating and inserting new block " << p
           << " from system allocator";

  static_cast<MemoryBlock*>(p)->init(&cache_, MemoryBlock::FREE_CHUNK, index,
                                     chunk_size, nullptr, nullptr);

  // gpu fallback allocation
  if (system_allocator_->UseGpu() &&
      static_cast<MemoryBlock*>(p)->index(cache_) == 1) {
    fallback_alloc_count_++;
  }

  total_free_ += chunk_size;

  // dump the block into pool
  return pool_.insert(IndexSizeAddress(index, chunk_size, p)).first;
}

BuddyAllocator::PoolSet::iterator BuddyAllocator::FindExistChunk(size_t size) {
  size_t index = 0;

  while (1) {
    auto it = pool_.lower_bound(IndexSizeAddress(index, size, nullptr));

    // no match chunk memory
    if (it == pool_.end()) return it;

    if (std::get<0>(*it) > index) {
      // find suitable one
      if (std::get<1>(*it) >= size) {
        return it;
      }
      // update and continue
      index = std::get<0>(*it);
      continue;
    }
    return it;
  }
}

void* BuddyAllocator::SplitToAlloc(BuddyAllocator::PoolSet::iterator it,
                                   size_t size) {
  auto block = static_cast<MemoryBlock*>(std::get<2>(*it));
  pool_.erase(it);

  VLOG(10) << "Split block (" << block << ", " << block->total_size(cache_)
           << ") into";
  block->split(&cache_, size);

  VLOG(10) << "Left block (" << block << ", " << block->total_size(cache_)
           << ")";
  block->set_type(&cache_, MemoryBlock::ARENA_CHUNK);

  // the rest of memory if exist
  if (block->has_right_buddy(cache_)) {
    if (block->right_buddy(cache_)->type(cache_) == MemoryBlock::FREE_CHUNK) {
      VLOG(10) << "Insert right block (" << block->right_buddy(cache_) << ", "
               << block->right_buddy(cache_)->total_size(cache_) << ")";

      pool_.insert(
          IndexSizeAddress(block->right_buddy(cache_)->index(cache_),
                           block->right_buddy(cache_)->total_size(cache_),
                           block->right_buddy(cache_)));
    }
  }

  return block;
}

void BuddyAllocator::CleanIdleFallBackAlloc() {
  // If fallback allocation does not exist, return directly
  if (!fallback_alloc_count_ || !system_allocator_->UseGpu()) return;

  for (auto pool = pool_.rbegin(); pool != pool_.rend();) {
    MemoryBlock* block = static_cast<MemoryBlock*>(std::get<2>(*pool));

    auto desc = cache_.load(block);
    if (desc.index == 0) {
      return;
    }

    VLOG(10) << "Return block " << block << " to fallback allocator.";

    system_allocator_->Free(block, desc.size, block->index(cache_));
    cache_.invalidate(block);

    pool = PoolSet::reverse_iterator(pool_.erase(std::next(pool).base()));

    total_free_ -= desc.size;
    fallback_alloc_count_--;

    // If no fall allocation exists, return directly
    if (!fallback_alloc_count_) return;
  }
}

void BuddyAllocator::CleanIdleNormalAlloc() {
  auto shall_free_alloc = [&]() -> bool {
    // free all fallback allocations
    if (fallback_alloc_count_ > 0) {
      return true;
    }
    // keep 2x overhead if we haven't fallen back
    if ((total_used_ + max_chunk_size_) * 2 < total_free_) {
      return true;
    }
    return false;
  };

  if (!shall_free_alloc()) return;

  for (auto pool = pool_.rbegin(); pool != pool_.rend();) {
    MemoryBlock* block = static_cast<MemoryBlock*>(std::get<2>(*pool));
    auto desc = cache_.load(block);

    if (desc.type != MemoryBlock::MANAGED_HUGE_CHUNK) {
      return;
    }

    VLOG(10) << "Return block " << block << " to base allocator.";

    system_allocator_->Free(block, desc.size, desc.index);
    cache_.invalidate(block);

    pool = PoolSet::reverse_iterator(pool_.erase(std::next(pool).base()));

    total_free_ -= desc.size;

    if (!shall_free_alloc()) return;
  }
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
