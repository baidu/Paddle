// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include "glog/logging.h"
#include "paddle/common/errors.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/utils/small_vector.h"

namespace forward_ad {

#define EXPECTED_MAX_LEVEL 2

struct ForwardGrad;
struct ForwardADLevel;

namespace {
// See discussion in forward_grad.h for why these are global variables and not
// thread local

// std::mutex all_forward_levels_mutex_;
// std::vector<std::shared_ptr<ForwardADLevel>> all_forward_levels_;

static const paddle::Tensor singleton_undefined_tensor;
}  // namespace

class ForwardADLevelManager {
 public:
  static ForwardADLevelManager& instance() {
    static ForwardADLevelManager instance;
    return instance;
  }

  std::vector<std::shared_ptr<ForwardADLevel>>& get_all_forward_levels() {
    return all_forward_levels_;
  }

 private:
  ForwardADLevelManager() = default;
  ~ForwardADLevelManager() = default;
  ForwardADLevelManager(const ForwardADLevelManager&) = delete;
  ForwardADLevelManager& operator=(const ForwardADLevelManager&) = delete;

  std::vector<std::shared_ptr<ForwardADLevel>> all_forward_levels_;
};

struct TEST_API ForwardADLevel {
  explicit ForwardADLevel(uint64_t idx) : idx_(idx) {}
  ~ForwardADLevel();

  static uint64_t get_next_idx() {
    // std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
    auto& all_forward_levels_ =
        ForwardADLevelManager::instance().get_all_forward_levels();
    auto next_idx = all_forward_levels_.size();
    PD_CHECK(next_idx == 0,
             "Nested forward mode AD is not supported at the moment");
    all_forward_levels_.push_back(std::make_shared<ForwardADLevel>(next_idx));
    // VLOG(0) << "address of all_forward_levels_ = " << (&all_forward_levels_);
    // VLOG(0) << "all_forward_levels_.size() = " << all_forward_levels_.size();
    return next_idx;
  }

  static void release_idx(uint64_t idx) {
    // std::unique_lock<std::mutex> lock(all_forward_levels_mutex_);
    auto& all_forward_levels_ =
        ForwardADLevelManager::instance().get_all_forward_levels();
    // VLOG(0) << "Call release_idx: " << idx;
    PD_CHECK(idx + 1 == all_forward_levels_.size(),
             "Exiting a forward AD level that is not the "
             "last that was created is not support. Ensure they are released "
             "in the reverse "
             "order they were created.");
    PD_CHECK(!all_forward_levels_.empty(),
             "all_forward_levels_ can not be empty.");
    // Keep the level alive until we have released the lock
    auto lvl = std::move(all_forward_levels_.back());
    all_forward_levels_.pop_back();
    // lock.unlock();
  }

  static std::shared_ptr<ForwardADLevel> get_by_idx(uint64_t idx) {
    // std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
    auto& all_forward_levels_ =
        ForwardADLevelManager::instance().get_all_forward_levels();
    // VLOG(0) << "idx = " << idx;
    // VLOG(0) << "address of all_forward_levels_ = " << (&all_forward_levels_);
    // VLOG(0) << "all_forward_levels_.size() = " << all_forward_levels_.size();
    PD_CHECK(idx < all_forward_levels_.size(),
             "Trying to access a forward AD level with an invalid index. "
             "This index was either not created or is already deleted.");
    return all_forward_levels_[idx];
  }

  static std::shared_ptr<ForwardADLevel> try_get_by_idx(uint64_t idx) {
    // std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
    auto& all_forward_levels_ =
        ForwardADLevelManager::instance().get_all_forward_levels();
    if (idx < all_forward_levels_.size()) {
      return all_forward_levels_[idx];
    } else {
      return nullptr;
    }
  }

  void erase(const std::shared_ptr<ForwardGrad>& grad) {
    // std::lock_guard<std::mutex> lock(mutex_);
    grads_.erase(grad);
  }

  void insert(const std::shared_ptr<ForwardGrad>& grad) {
    // std::lock_guard<std::mutex> lock(mutex_);
    grads_.insert(grad);
  }

 private:
  std::unordered_set<std::shared_ptr<ForwardGrad>>
      grads_;  // 存储ForwardGrad的集合
  // std::mutex mutex_;
  uint64_t idx_;
};

struct TEST_API ForwardGrad : std::enable_shared_from_this<ForwardGrad> {
  ForwardGrad() = default;

  // This function must only be called when AutogradMeta or SavedVariable is
  // being destructed as it ensures that:
  //   - The only (potential) other references to this ForwardGrad are the
  //     different level it is registered to
  //   - No other thread will try to call `set_value` or `value` ever from now
  //   on
  //   - Any of the ForwardADLevel that this ForwardGrad is registered with
  //   might
  //     call `reset` at any point during this function
  void clear() {
    paddle::small_vector<uint64_t, EXPECTED_MAX_LEVEL> levels_idx;

    {
      // std::lock_guard<std::mutex> lock(mutex_);
      for (auto& c : content_) {
        levels_idx.push_back(c.first);
      }
    }

    for (auto l_idx : levels_idx) {
      // Use "try" version here as another thread might have deleted this
      // level before we got here
      // This is an owning reference as we want to keep the level alive
      // until we successfully unregister ourselves
      auto level = ForwardADLevel::try_get_by_idx(l_idx);
      if (level) {
        level->erase(shared_from_this());
      }
    }
  }

  void set_value(const paddle::Tensor& value, uint64_t level) {
    // Owning reference to ensure the forward_level is not destroyed
    // while we are updating our internal state
    auto forward_level = ForwardADLevel::get_by_idx(level);
    forward_level->insert(shared_from_this());

    // std::lock_guard<std::mutex> lock(mutex_);
    content_.insert({level, value});
  }

  // This function removes the tangent for a given level from this ForwardGrad
  // Use the update_level flag to disable notifying the level about this reset
  // This flag is most notably used by the ForwardADLevel destructor.
  // 将一个Tensor的某个level的fwdgrad从content_中移除，并从ForwardADLevel中删除该level的引用
  void reset(uint64_t level, bool update_level = true) {
    if (update_level) {  // if
                         // update_level，那么甚至需要从ForwardADLevel中删除本ForwardGrad的引用
      ForwardADLevel::get_by_idx(level)->erase(shared_from_this());
    }

    // std::unique_lock<std::mutex> lock(mutex_);
    const auto& it = content_.find(level);
    PD_CHECK(it != content_.end(), "Resetting a non-existent level.");
    // Keep the Tensor alive until we have released the lock
    // This is needed as we can be in a case where this function is called by
    // ForwardADLevel destructor
    auto t = (*it).second;
    content_.erase(level);
    // lock.unlock();
  }

  const paddle::Tensor& value(uint64_t level) const {
    // std::lock_guard<std::mutex> lock(mutex_);
    const auto& it = content_.find(level);
    return it == content_.end() ? singleton_undefined_tensor : (*it).second;
  }

  bool contains(uint64_t level) {
    // std::lock_guard<std::mutex> lock(mutex_);
    return content_.count(level) > 0;
  }

  bool empty() const { return content_.empty(); }

  static const paddle::Tensor& undef_grad() {
    return singleton_undefined_tensor;
  }

 private:
  // TODO(albanD): replace this with a SmallVector
  // 每个Tensor都有一个ForwardGrad对象，用来存储不同level的（前向）梯度，所以是level->Tensor的映射
  std::unordered_map<uint64_t, paddle::Tensor> content_;
  // mutable std::mutex mutex_; // 进程间互斥锁
};

}  // namespace forward_ad
