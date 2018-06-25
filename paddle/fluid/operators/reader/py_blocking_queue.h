//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <condition_variable>  // NOLINT
#include <deque>

#include "Python.h"
#include "paddle/fluid/platform/enforce.h"
#include "pybind11/pybind11.h"

namespace paddle {
namespace operators {
namespace reader {

// PyBlockingQueue is designed for PyArrayFeedQueue
// PyBlockingQueue would release GIL of Python when
// the queue is full to avoid deadlock.
template <typename T>
class PyBlockingQueue {
 public:
  explicit PyBlockingQueue(size_t capacity)
      : capacity_(capacity), closed_(false) {
    PADDLE_ENFORCE_GT(
        capacity_, 0,
        "The capacity of a reader::PyBlockingQueue must be greater than 0.");
  }

  ~PyBlockingQueue() { Close(); }

  bool Send(const T& elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    receive_cv_.notify_one();
    if (queue_.size() >= capacity_ && (!closed_)) {
      pybind11::gil_scoped_release release;
      send_cv_.wait(lock, [&] { return queue_.size() < capacity_ || closed_; });
    }
    if (closed_) {
      VLOG(5)
          << "WARNING: Sending an element to a closed reader::BlockingQueue.";
      return false;
    }
    PADDLE_ENFORCE_LT(queue_.size(), capacity_);
    queue_.push_back(elem);
    return true;
  }

  bool Send(T&& elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    receive_cv_.notify_one();
    if (queue_.size() >= capacity_ && (!closed_)) {
      pybind11::gil_scoped_release release;
      send_cv_.wait(lock, [&] { return queue_.size() < capacity_ || closed_; });
    }
    if (closed_) {
      VLOG(5)
          << "WARNING: Sending an element to a closed reader::BlokcingQueue.";
      return false;
    }
    PADDLE_ENFORCE_LT(queue_.size(), capacity_);
    queue_.emplace_back(std::move(elem));
    return true;
  }

  bool Receive(T* elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    send_cv_.notify_one();
    receive_cv_.wait(lock, [&] { return !queue_.empty() || closed_; });
    if (!queue_.empty()) {
      PADDLE_ENFORCE_NOT_NULL(elem);
      *elem = queue_.front();
      queue_.pop_front();
      return true;
    } else {
      PADDLE_ENFORCE(closed_);
      return false;
    }
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    send_cv_.notify_all();
    receive_cv_.notify_all();
  }

  bool IsClosed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

  size_t Cap() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  size_t capacity_;
  bool closed_;
  std::deque<T> queue_;

  mutable std::mutex mutex_;
  mutable std::condition_variable receive_cv_;
  mutable std::condition_variable send_cv_;
};
}  // namespace reader
}  // namespace operators
}  // namespace paddle
