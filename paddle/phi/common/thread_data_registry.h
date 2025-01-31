// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>

namespace phi {

#ifdef __cpp_lib_void_t
using std::void_t;
#else
template <typename...>
using void_t = void;
#endif

template <typename T, typename = void>
struct IsAccumulatable : std::false_type {};

template <>
struct IsAccumulatable<std::string, void> : std::false_type {};

template <typename T>
struct IsAccumulatable<T,
                       void_t<decltype(std::declval<T>() += std::declval<T>())>>
    : std::true_type {};

template <typename T>
class ThreadDataRegistry {
 public:
  // Singleton
  static ThreadDataRegistry& GetInstance() {
    static ThreadDataRegistry instance;
    return instance;
  }

  T* GetMutableCurrentThreadData() { return &CurrentThreadData(); }

  const T& GetCurrentThreadData() { return CurrentThreadData(); }

  template <typename Alias = T,
            typename = std::enable_if_t<std::is_copy_assignable<Alias>::value>>
  void SetCurrentThreadData(const T& val) {
    CurrentThreadData() = val;
  }

  // Returns current snapshot of all threads. Make sure there is no thread
  // create/destroy when using it.
  template <
      typename Alias = T,
      typename = std::enable_if_t<std::is_copy_constructible<Alias>::value>>
  std::unordered_map<uint64_t, T> GetAllThreadDataByValue() {
    return impl_->GetAllThreadDataByValue();
  }

  // Returns current snapshot of all threads. Make sure there is no thread
  // create/destroy when using it.
  std::unordered_map<uint64_t, std::reference_wrapper<T>>
  GetAllThreadDataByRef() {
    return impl_->GetAllThreadDataByRef();
  }

 private:
// types
// Lock types
#if defined(__clang__) || defined(__GNUC__)  // CLANG or GCC
#ifndef __APPLE__
#if __cplusplus >= 201703L
  using LockType = std::shared_mutex;
  using SharedLockGuardType = std::shared_lock<std::shared_mutex>;
#elif __cplusplus >= 201402L
  using LockType = std::shared_timed_mutex;
  using SharedLockGuardType = std::shared_lock<std::shared_timed_mutex>;
#else
  using LockType = std::mutex;
  using SharedLockGuardType = std::lock_guard<std::mutex>;
#endif
// Special case : mac. https://github.com/facebook/react-native/issues/31250
#else
  using LockType = std::mutex;
  using SharedLockGuardType = std::lock_guard<std::mutex>;
#endif
#elif defined(_MSC_VER)  // MSVC
#if _MSVC_LANG >= 201703L
  using LockType = std::shared_mutex;
  using SharedLockGuardType = std::shared_lock<std::shared_mutex>;
#elif _MSVC_LANG >= 201402L
  using LockType = std::shared_timed_mutex;
  using SharedLockGuardType = std::shared_lock<std::shared_timed_mutex>;
#else
  using LockType = std::mutex;
  using SharedLockGuardType = std::lock_guard<std::mutex>;
#endif
#else  // other compilers
  using LockType = std::mutex;
  using SharedLockGuardType = std::lock_guard<std::mutex>;
#endif

  class ThreadDataHolder;
  class ThreadDataRegistryImpl {
   public:
    void RegisterData(uint64_t tid, ThreadDataHolder* tls_obj) {
      std::lock_guard<LockType> guard(lock_);
      tid_map_[tid] = tls_obj;
    }

    template <typename Alias = T>
    void AccumulateToAnotherThread(...) {}

    template <typename Alias = T,
              typename = std::enable_if_t<IsAccumulatable<Alias>::value>>
    void AccumulateToAnotherThread(uint64_t tid) {
      if (tid_map_.find(tid) == tid_map_.end()) {
        return;
      }

      auto& data = tid_map_[tid]->GetData();
      for (auto& kv : tid_map_) {
        if (kv.first != tid) {
          auto& data_in_another_thread = kv.second->GetData();
          data_in_another_thread += data;
          VLOG(2) << "Add data " << data << " from thread " << tid << " to "
                  << kv.first << " , after update, data is "
                  << data_in_another_thread << ".";
          break;
        }
      }
    }

    void UnregisterData(uint64_t tid) {
      std::lock_guard<LockType> guard(lock_);
      AccumulateToAnotherThread(tid);
      tid_map_.erase(tid);
    }

    template <
        typename Alias = T,
        typename = std::enable_if_t<std::is_copy_constructible<Alias>::value>>
    std::unordered_map<uint64_t, T> GetAllThreadDataByValue() {
      std::unordered_map<uint64_t, T> data_copy;
      SharedLockGuardType guard(lock_);
      data_copy.reserve(tid_map_.size());
      for (auto& kv : tid_map_) {
        data_copy.emplace(kv.first, kv.second->GetData());
      }
      return data_copy;
    }

    std::unordered_map<uint64_t, std::reference_wrapper<T>>
    GetAllThreadDataByRef() {
      std::unordered_map<uint64_t, std::reference_wrapper<T>> data_ref;
      SharedLockGuardType guard(lock_);
      data_ref.reserve(tid_map_.size());
      for (auto& kv : tid_map_) {
        data_ref.emplace(kv.first, std::ref(kv.second->GetData()));
      }
      return data_ref;
    }

   private:
    LockType lock_;
    std::unordered_map<uint64_t, ThreadDataHolder*> tid_map_;  // not owned
  };

  class ThreadDataHolder {
   public:
    explicit ThreadDataHolder(
        std::shared_ptr<ThreadDataRegistryImpl> registry) {
      registry_ = std::move(registry);
      tid_ = std::hash<std::thread::id>()(std::this_thread::get_id());
      registry_->RegisterData(tid_, this);
    }

    ~ThreadDataHolder() { registry_->UnregisterData(tid_); }

    T& GetData() { return data_; }

   private:
    std::shared_ptr<ThreadDataRegistryImpl> registry_;
    uint64_t tid_;
    T data_;
  };

  // methods
  ThreadDataRegistry() { impl_ = std::make_shared<ThreadDataRegistryImpl>(); }

  ThreadDataRegistry(const ThreadDataRegistry&) = delete;

  ThreadDataRegistry& operator=(const ThreadDataRegistry&) = delete;

  T& CurrentThreadData() {
    static thread_local ThreadDataHolder thread_data(impl_);
    return thread_data.GetData();
  }

  // data
  std::shared_ptr<ThreadDataRegistryImpl> impl_;
};

}  // namespace phi
