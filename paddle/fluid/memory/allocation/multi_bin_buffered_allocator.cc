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

#include "paddle/fluid/memory/allocation/multi_bin_buffered_allocator.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include "paddle/fluid/platform/lock_guard_ptr.h"

DEFINE_double(buffered_allocator_excess_times, 2,
              "Tolerant memory size times of buffered_allocator");

DEFINE_string(division_plan_path, "", "Division plan file path");

namespace paddle {
namespace memory {
namespace allocation {

std::string TrimStringAndToLowerCase(const std::string &str) {
  auto not_space = [](char ch) { return std::isspace(ch) == 0; };
  auto first_idx = static_cast<size_t>(
      std::find_if(str.begin(), str.end(), not_space) - str.begin());
  auto last_idx = static_cast<size_t>(
      std::find_if(str.rbegin(), str.rend(), not_space) - str.rbegin());
  if (first_idx == str.size() || last_idx == str.size()) return "";

  last_idx = str.size() - 1 - last_idx;
  auto ret = str.substr(first_idx, last_idx - first_idx);
  std::for_each(ret.begin(), ret.end(),
                [](char &ch) { ch = std::tolower(ch); });
  return ret;
}

static size_t ParseStringToBytes(const std::string &str) {
  std::string ret = str;
  if (ret.back() == 'b') {
    ret.pop_back();
  }

  PADDLE_ENFORCE(!ret.empty(), "Wrong format: %s", str);
  size_t multiples = 1;
  switch (ret.back()) {
    case 'g':
      multiples *= (static_cast<size_t>(1) << 30);
      break;
    case 'm':
      multiples *= (static_cast<size_t>(1) << 20);
      break;
    case 'k':
      multiples *= (static_cast<size_t>(1) << 10);
      break;
    default:
      break;
  }

  if (multiples != 1) ret.pop_back();
  ret = TrimStringAndToLowerCase(ret);
  double ret_val = 0.0;
  std::stringstream ss(ret);
  PADDLE_ENFORCE((ss >> ret_val).good(), "Wrong format %s", str);
  return static_cast<size_t>(ret_val * multiples);
}

static std::string GetDebugStringOfPlan(const std::vector<size_t> &plan) {
  std::string ret("[");
  for (auto sz : plan) {
    ret += string::HumanReadableSize(sz);
    ret += ", ";
  }
  return ret + "]";
}

static std::vector<size_t> ReadDivisionPlanFromFile(
    const std::string &filepath) {
  std::ifstream is(filepath.c_str());
  PADDLE_ENFORCE(is.good(), "File not exist");
  std::string str;
  std::vector<size_t> plan;
  while (std::getline(is, str).good()) {
    str = TrimStringAndToLowerCase(str);
    if (str.empty()) break;
    plan.push_back(ParseStringToBytes(str));
  }
  return plan;
}

static void CheckAndModifyMemoryDivisionPlan(
    std::vector<size_t> *division_plan) {
  // Check whether the division plan is strictly sorted
  bool is_strictly_sorted = true;
  for (size_t i = 1; i < division_plan->size(); ++i) {
    if ((*division_plan)[i - 1] >= (*division_plan)[i]) {
      is_strictly_sorted = false;
      break;
    }
  }
  PADDLE_ENFORCE(is_strictly_sorted, "Divison plan must be stricted sorted");

  // Insert 0 and remove MAX to disivion plan for clean binary searching code
  if (division_plan->empty() || division_plan->front() != 0) {
    division_plan->insert(division_plan->begin(), 0);
  }

  constexpr auto kSizeTypeMax = std::numeric_limits<size_t>::max();
  if (division_plan->back() == kSizeTypeMax) {
    division_plan->pop_back();
  }

  PADDLE_ENFORCE(division_plan->size() >= 1, "Division plan cannot be empty");
}

static std::vector<size_t> GetDefaultDivisionPlan() {
  if (!FLAGS_division_plan_path.empty()) {
    return ReadDivisionPlanFromFile(FLAGS_division_plan_path);
  }

  constexpr size_t kMaxLogSize = 30;

  std::vector<size_t> plan;
  for (size_t i = 12; i <= kMaxLogSize; ++i) {
    plan.push_back(static_cast<size_t>(1) << i);
  }
  /*
  for (size_t i = 0; i < sizeof(size_t) * 8; ++i) {
    plan.push_back(static_cast<size_t>(1) << i);
  }
  */
  return plan;
}

inline static size_t FindDivisionPlanBinIndex(const std::vector<size_t> &bins,
                                              size_t size) {
  return static_cast<size_t>(std::upper_bound(bins.begin(), bins.end(), size) -
                             bins.begin() - 1);
}

inline static size_t TolerantUpperSize(size_t size) {
  return static_cast<size_t>(size * FLAGS_buffered_allocator_excess_times);
}

MultiBinBufferedAllocator::MultiBinBufferedAllocator(
    std::shared_ptr<Allocator> underlying_allocator)
    : MultiBinBufferedAllocator(std::move(underlying_allocator),
                                GetDefaultDivisionPlan()) {}

MultiBinBufferedAllocator::MultiBinBufferedAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    const std::vector<size_t> &division_plan)
    : underlying_allocator_(std::move(underlying_allocator)),
      division_plan_(division_plan) {
  CheckAndModifyMemoryDivisionPlan(&division_plan_);
  allocations_.resize(division_plan_.size() - 1);
  mtx_.resize(division_plan_.size() - 1);
  if (underlying_allocator_->IsAllocThreadSafe()) {
    for (auto &mtx : mtx_) {
      mtx.reset(new std::mutex());
    }
  }

  VLOG(1) << "Division plan is: " << GetDebugStringOfPlan(division_plan_);
  VLOG(1) << "FLAGS_buffered_allocator_excess_times = "
          << FLAGS_buffered_allocator_excess_times;
}

void MultiBinBufferedAllocator::FreeImpl(Allocation *allocation) {
  auto bin_index = FindDivisionPlanBinIndex(division_plan_, allocation->size());
  if (bin_index < allocations_.size()) {
    platform::LockGuardPtr<std::mutex> guard(mtx_[bin_index]);
    allocations_[bin_index].emplace(allocation->size(),
                                    AllocationPtr(allocation));
  } else {
    underlying_allocator_->Free(allocation);
  }
}

// bin_index is not used currently.
// Maybe we can design more flexible FreeCache strategy based on bin_index
size_t MultiBinBufferedAllocator::FreeCache(size_t size, size_t bin_index) {
  size_t accumulated_size = 0;
  // FIXME(zjl): free the largest first when there is no extra
  for (size_t i = allocations_.size() - 1; i != static_cast<size_t>(-1); --i) {
    platform::LockGuardPtr<std::mutex> lock(mtx_[i]);
    if (allocations_[i].empty()) continue;
    auto it = --allocations_[i].end();
    do {
      accumulated_size += it->second->size();
      underlying_allocator_->Free(it->second.release());
      allocations_[i].erase(it--);
      if (accumulated_size >= size) {
        return accumulated_size;
      }
    } while (!allocations_[i].empty());
  }
  return accumulated_size;
}

Allocation *MultiBinBufferedAllocator::AllocateImpl(size_t size, Attr attr) {
  auto bin_index = FindDivisionPlanBinIndex(division_plan_, size);
  auto upper_size = TolerantUpperSize(size);

  // if (bin_index >= allocations_.size()) {
  //  VLOG(2) << "Allocate " << size << " from underlying directly";
  //}

  for (; bin_index < allocations_.size() &&
         upper_size >= division_plan_[bin_index];
       ++bin_index) {
    auto &allocation = allocations_[bin_index];
    platform::LockGuardPtr<std::mutex> lock(mtx_[bin_index]);
    auto it = allocation.lower_bound(size);
    if (it != allocation.end() && it->second->size() <= upper_size) {
      size_t sz = it->second->size();
      auto ret = std::move(it->second);
      allocation.erase(it);
      VLOG(3) << "Allocate " << sz << "(required " << size
              << ") from cache directly";
      return ret.release();
    }
  }

  size_t retry_time = 1;
  while (true) {
    try {
      auto ret = underlying_allocator_->Allocate(size, attr).release();
      VLOG(2) << "Allocate " << size << " from underlying directly";
      return ret;
    } catch (BadAlloc &) {
      VLOG(1) << retry_time << "-th BadAlloc raises, try to free " << size
              << " bytes caches";
      // size_t actual_free_size = FreeCache(size, bin_index);
      size_t actual_free_size = FreeCache(-1UL, bin_index);
      VLOG(1) << retry_time << "-th free " << actual_free_size
              << " bytes caches";
      if (actual_free_size == 0) throw;
    }
    ++retry_time;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
