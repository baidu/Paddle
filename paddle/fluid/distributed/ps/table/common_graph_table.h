// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <ThreadPool.h>
#include <assert.h>
#include <pthread.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <ctime>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/common_table.h"
#include "paddle/fluid/distributed/ps/table/graph/class_macro.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/distributed/ps/thirdparty/round_robin.h"
#include "paddle/phi/core/utils/rw_lock.h"
#include "paddle/utils/string/string_helper.h"

#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/distributed/ps/table/depends/rocksdb_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#endif
namespace paddle {
namespace distributed {
class GraphShard {
 public:
  size_t get_size();
  GraphShard() {}
  ~GraphShard();
  std::vector<Node *> &get_bucket() { return bucket; }
  std::vector<Node *> get_batch(int start, int end, int step);
  void get_ids_by_range(int start, int end, std::vector<uint64_t> *res) {
    res->reserve(res->size() + end - start);
    for (int i = start; i < end && i < static_cast<int>(bucket.size()); i++) {
      res->emplace_back(bucket[i]->get_id());
    }
  }
  size_t get_all_id(std::vector<std::vector<uint64_t>> *shard_keys,
                    int slice_num) {
    int bucket_num = bucket.size();
    shard_keys->resize(slice_num);
    for (int i = 0; i < slice_num; ++i) {
      (*shard_keys)[i].reserve(bucket_num / slice_num);
    }
    for (int i = 0; i < bucket_num; i++) {
      uint64_t k = bucket[i]->get_id();
      (*shard_keys)[k % slice_num].emplace_back(k);
    }
    return bucket_num;
  }
  size_t get_all_neighbor_id(std::vector<std::vector<uint64_t>> *total_res,
                             int slice_num) {
    std::vector<uint64_t> keys;
    for (size_t i = 0; i < bucket.size(); i++) {
      size_t neighbor_size = bucket[i]->get_neighbor_size();
      size_t n = keys.size();
      keys.resize(n + neighbor_size);
      for (size_t j = 0; j < neighbor_size; j++) {
        keys[n + j] = bucket[i]->get_neighbor_id(j);
      }
    }
    return dedup2shard_keys(&keys, total_res, slice_num);
  }
  size_t get_all_feature_ids(std::vector<std::vector<uint64_t>> *total_res,
                             int slice_num) {
    std::vector<uint64_t> keys;
    for (size_t i = 0; i < bucket.size(); i++) {
      bucket[i]->get_feature_ids(&keys);
    }
    return dedup2shard_keys(&keys, total_res, slice_num);
  }
  size_t dedup2shard_keys(std::vector<uint64_t> *keys,
                          std::vector<std::vector<uint64_t>> *total_res,
                          int slice_num) {
    size_t num = keys->size();
    uint64_t last_key = 0;
    // sort key insert to vector
    std::sort(keys->begin(), keys->end());
    total_res->resize(slice_num);
    for (int shard_id = 0; shard_id < slice_num; ++shard_id) {
      (*total_res)[shard_id].reserve(num / slice_num);
    }
    for (size_t i = 0; i < num; ++i) {
      const uint64_t &k = (*keys)[i];
      if (i > 0 && last_key == k) {
        continue;
      }
      last_key = k;
      (*total_res)[k % slice_num].push_back(k);
    }
    return num;
  }
  GraphNode *add_graph_node(uint64_t id);
  GraphNode *add_graph_node(Node *node);
  FeatureNode *add_feature_node(uint64_t id,
                                bool is_overlap = true,
                                int float_fea_num = 0);
  Node *find_node(uint64_t id);
  void delete_node(uint64_t id);
  void clear();
  void add_neighbor(uint64_t id, uint64_t dst_id, float weight);
  std::unordered_map<uint64_t, int> &get_node_location() {
    return node_location;
  }

  void shrink_to_fit() {
    bucket.shrink_to_fit();
    for (size_t i = 0; i < bucket.size(); i++) {
      bucket[i]->shrink_to_fit();
    }
  }

  void merge_shard(GraphShard *&shard) {  // NOLINT
    bucket.reserve(bucket.size() + shard->bucket.size());
    for (size_t i = 0; i < shard->bucket.size(); i++) {
      auto node_id = shard->bucket[i]->get_id();
      if (node_location.find(node_id) == node_location.end()) {
        node_location[node_id] = bucket.size();
        bucket.push_back(shard->bucket[i]);
      }
    }
    shard->node_location.clear();
    shard->bucket.clear();
    delete shard;
    shard = NULL;
  }

 public:
  std::unordered_map<uint64_t, int> node_location;
  std::vector<Node *> bucket;
};

enum LRUResponse { ok = 0, blocked = 1, err = 2 };

struct SampleKey {
  int idx;
  uint64_t node_key;
  size_t sample_size;
  bool is_weighted;
  SampleKey(int _idx,
            uint64_t _node_key,
            size_t _sample_size,
            bool _is_weighted) {
    idx = _idx;
    node_key = _node_key;
    sample_size = _sample_size;
    is_weighted = _is_weighted;
  }
  bool operator==(const SampleKey &s) const {
    return idx == s.idx && node_key == s.node_key &&
           sample_size == s.sample_size && is_weighted == s.is_weighted;
  }
};

class SampleResult {
 public:
  size_t actual_size;
  std::shared_ptr<char> buffer;
  SampleResult(size_t _actual_size, std::shared_ptr<char> &_buffer)  // NOLINT
      : actual_size(_actual_size), buffer(_buffer) {}
  SampleResult(size_t _actual_size, char *_buffer)
      : actual_size(_actual_size),
        buffer(_buffer, [](char *p) { delete[] p; }) {}
  ~SampleResult() {}
};

template <typename K, typename V>
class LRUNode {
 public:
  LRUNode(K _key, V _data, size_t _ttl) : key(_key), data(_data), ttl(_ttl) {
    next = pre = NULL;
  }
  K key;
  V data;
  size_t ttl;
  // time to live
  LRUNode<K, V> *pre, *next;
};
template <typename K, typename V>
class ScaledLRU;

template <typename K, typename V>
class RandomSampleLRU {
 public:
  explicit RandomSampleLRU(ScaledLRU<K, V> *_father) {
    father = _father;
    remove_count = 0;
    node_size = 0;
    node_head = node_end = NULL;
    global_ttl = father->ttl;
    total_diff = 0;
  }

  ~RandomSampleLRU() {
    LRUNode<K, V> *p;
    while (node_head != NULL) {
      p = node_head->next;
      delete node_head;
      node_head = p;
    }
  }
  LRUResponse query(K *keys,
                    size_t length,
                    std::vector<std::pair<K, V>> &res) {  // NOLINT
    if (pthread_rwlock_tryrdlock(&father->rwlock) != 0)
      return LRUResponse::blocked;
    // pthread_rwlock_rdlock(&father->rwlock);
    int init_size = node_size - remove_count;
    process_redundant(length * 3);

    for (size_t i = 0; i < length; i++) {
      auto iter = key_map.find(keys[i]);
      if (iter != key_map.end()) {
        res.emplace_back(keys[i], iter->second->data);
        iter->second->ttl--;
        if (iter->second->ttl == 0) {
          remove(iter->second);
          if (remove_count != 0) remove_count--;
        } else {
          move_to_tail(iter->second);
        }
      }
    }
    total_diff += node_size - remove_count - init_size;
    if (total_diff >= 500 || total_diff < -500) {
      father->handle_size_diff(total_diff);
      total_diff = 0;
    }
    pthread_rwlock_unlock(&father->rwlock);
    return LRUResponse::ok;
  }
  LRUResponse insert(K *keys, V *data, size_t length) {
    if (pthread_rwlock_tryrdlock(&father->rwlock) != 0)
      return LRUResponse::blocked;
    // pthread_rwlock_rdlock(&father->rwlock);
    int init_size = node_size - remove_count;
    process_redundant(length * 3);
    for (size_t i = 0; i < length; i++) {
      auto iter = key_map.find(keys[i]);
      if (iter != key_map.end()) {
        move_to_tail(iter->second);
        iter->second->ttl = global_ttl;
        iter->second->data = data[i];
      } else {
        LRUNode<K, V> *temp = new LRUNode<K, V>(keys[i], data[i], global_ttl);
        add_new(temp);
      }
    }
    total_diff += node_size - remove_count - init_size;
    if (total_diff >= 500 || total_diff < -500) {
      father->handle_size_diff(total_diff);
      total_diff = 0;
    }

    pthread_rwlock_unlock(&father->rwlock);
    return LRUResponse::ok;
  }
  void remove(LRUNode<K, V> *node) {
    fetch(node);
    node_size--;
    key_map.erase(node->key);
    delete node;
  }

  void process_redundant(int process_size) {
    int length = std::min(remove_count, process_size);
    while (length--) {
      remove(node_head);
      remove_count--;
    }
  }

  void move_to_tail(LRUNode<K, V> *node) {
    fetch(node);
    place_at_tail(node);
  }

  void add_new(LRUNode<K, V> *node) {
    node->ttl = global_ttl;
    place_at_tail(node);
    node_size++;
    key_map[node->key] = node;
  }
  void place_at_tail(LRUNode<K, V> *node) {
    if (node_end == NULL) {
      node_head = node_end = node;
      node->next = node->pre = NULL;
    } else {
      node_end->next = node;
      node->pre = node_end;
      node->next = NULL;
      node_end = node;
    }
  }

  void fetch(LRUNode<K, V> *node) {
    if (node->pre) {
      node->pre->next = node->next;
    } else {
      node_head = node->next;
    }
    if (node->next) {
      node->next->pre = node->pre;
    } else {
      node_end = node->pre;
    }
  }

 private:
  std::unordered_map<K, LRUNode<K, V> *> key_map;
  ScaledLRU<K, V> *father;
  size_t global_ttl, size_limit;
  int node_size, total_diff;
  LRUNode<K, V> *node_head, *node_end;
  friend class ScaledLRU<K, V>;
  int remove_count;
};

template <typename K, typename V>
class ScaledLRU {
 public:
  ScaledLRU(size_t _shard_num, size_t size_limit, size_t _ttl)
      : size_limit(size_limit), ttl(_ttl) {
    shard_num = _shard_num;
    pthread_rwlock_init(&rwlock, NULL);
    stop = false;
    thread_pool.reset(new ::ThreadPool(1));
    global_count = 0;
    lru_pool = std::vector<RandomSampleLRU<K, V>>(shard_num,
                                                  RandomSampleLRU<K, V>(this));
    shrink_job = std::thread([this]() -> void {
      while (true) {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait_for(lock, std::chrono::milliseconds(20000));
          if (stop) {
            return;
          }
        }
        auto status =
            thread_pool->enqueue([this]() -> int { return Shrink(); });
        status.wait();
      }
    });
    shrink_job.detach();
  }
  ~ScaledLRU() {
    std::unique_lock<std::mutex> lock(mutex_);
    stop = true;
    cv_.notify_one();
  }
  LRUResponse query(size_t index,
                    K *keys,
                    size_t length,
                    std::vector<std::pair<K, V>> &res) {  // NOLINT
    return lru_pool[index].query(keys, length, res);
  }
  LRUResponse insert(size_t index, K *keys, V *data, size_t length) {
    return lru_pool[index].insert(keys, data, length);
  }
  int Shrink() {
    size_t node_size = 0;
    for (size_t i = 0; i < lru_pool.size(); i++) {
      node_size += lru_pool[i].node_size - lru_pool[i].remove_count;
    }

    if (node_size <= static_cast<size_t>(1.1 * size_limit) + 1) return 0;
    if (pthread_rwlock_wrlock(&rwlock) == 0) {
      global_count = 0;
      for (size_t i = 0; i < lru_pool.size(); i++) {
        global_count += lru_pool[i].node_size - lru_pool[i].remove_count;
      }
      if (static_cast<size_t>(global_count) > size_limit) {
        size_t remove = global_count - size_limit;
        for (size_t i = 0; i < lru_pool.size(); i++) {
          lru_pool[i].total_diff = 0;
          lru_pool[i].remove_count +=
              1.0 * (lru_pool[i].node_size - lru_pool[i].remove_count) /
              global_count * remove;
        }
      }
      pthread_rwlock_unlock(&rwlock);
      return 0;
    }
    return 0;
  }

  void handle_size_diff(int diff) {
    if (diff != 0) {
      __sync_fetch_and_add(&global_count, diff);
      if (global_count > static_cast<int>(1.25 * size_limit)) {
        thread_pool->enqueue([this]() -> int { return Shrink(); });
      }
    }
  }

  size_t get_ttl() { return ttl; }

 private:
  pthread_rwlock_t rwlock;
  size_t shard_num;
  int global_count;
  size_t size_limit, total, hit;
  size_t ttl;
  bool stop;
  std::thread shrink_job;
  std::vector<RandomSampleLRU<K, V>> lru_pool;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::shared_ptr<::ThreadPool> thread_pool;
  friend class RandomSampleLRU<K, V>;
};
enum GraphTableType { EDGE_TABLE, FEATURE_TABLE, NODE_TABLE };
class GraphTable : public Table {
  class GraphNodeRank {
   public:
    GraphNodeRank() {}
    ~GraphNodeRank() {}
    void init(const int node_num, const int shard_num) {
      shard_num_ = shard_num;
      node_num_ = node_num;
      rank_nodes_.resize(node_num);
      rank_sizes_.resize(node_num, 0);
      for (int i = 0; i < node_num_; ++i) {
        rank_nodes_[i].resize(shard_num_);
      }
    }
    void clear(void) {
      rank_nodes_.clear();
      rank_nodes_.shrink_to_fit();
    }
    void rehash(const int &shard_id, const size_t count) {
      size_t per_count = (count + node_num_ - 1) / node_num_;
      for (int i = 0; i < node_num_; ++i) {
        rank_nodes_[i][shard_id].rehash(per_count);
      }
    }
    bool empty(void) { return rank_nodes_.empty(); }
    const size_t &nodes_num(const int &rank) { return rank_sizes_[rank]; }
    bool add(const uint64_t &key, const int &rank) {
      auto &hash = rank_nodes_[rank][key % shard_num_];
      auto it = hash.find(key);
      if (it != hash.end()) {
        return false;
      }
      hash.insert(key);
      ++rank_sizes_[rank];
      return true;
    }
    void insert(const uint64_t &key, const int &rank) {
      rank_nodes_[rank][key % shard_num_].insert(key);
      ++rank_sizes_[rank];
    }
    int find(const uint64_t &key) {
      int shard_id = key % shard_num_;
      for (int i = 0; i < node_num_; ++i) {
        auto &hash = rank_nodes_[i][shard_id];
        auto it = hash.find(key);
        if (it == hash.end()) {
          continue;
        }
        return i;
      }
      return -1;
    }
    std::vector<std::vector<robin_hood::unordered_set<uint64_t>>>
    get_rank_nodes() {
      return rank_nodes_;
    }

   private:
    int node_num_ = -1;
    int shard_num_ = -1;
    std::vector<std::vector<robin_hood::unordered_set<uint64_t>>> rank_nodes_;
    std::vector<size_t> rank_sizes_;
  };

 public:
  GraphTable() {
    use_cache = false;
    shard_num = 0;
    rw_lock.reset(new pthread_rwlock_t());
#ifdef PADDLE_WITH_HETERPS
    next_partition = 0;
    total_memory_cost = 0;
#endif
  }
  virtual ~GraphTable();

  virtual void *GetShard(size_t shard_idx UNUSED) { return 0; }

  static int32_t sparse_local_shard_num(uint32_t shard_num,
                                        uint32_t server_num) {
    if (shard_num % server_num == 0) {
      return shard_num / server_num;
    }
    size_t local_shard_num = shard_num / server_num + 1;
    return local_shard_num;
  }

  static size_t get_sparse_shard(uint32_t shard_num,
                                 uint32_t server_num,
                                 uint64_t key) {
    return (key % shard_num) / sparse_local_shard_num(shard_num, server_num);
  }

  virtual int32_t pull_graph_list(GraphTableType table_type,
                                  int idx,
                                  int start,
                                  int size,
                                  std::unique_ptr<char[]> &buffer,  // NOLINT
                                  int &actual_size,                 // NOLINT
                                  bool need_feature,
                                  int step);

  virtual int32_t random_sample_neighbors(
      int idx,
      uint64_t *node_ids,
      int sample_size,
      std::vector<std::shared_ptr<char>> &buffers,  // NOLINT
      std::vector<int> &actual_sizes,               // NOLINT
      bool need_weight);

  int32_t random_sample_nodes(GraphTableType table_type,
                              int idx,
                              int sample_size,
                              std::unique_ptr<char[]> &buffers,  // NOLINT
                              int &actual_sizes);                // NOLINT

  virtual int32_t get_nodes_ids_by_ranges(
      GraphTableType table_type,
      int idx,
      std::vector<std::pair<int, int>> ranges,
      std::vector<uint64_t> &res);  // NOLINT
  virtual int32_t Initialize() { return 0; }
  virtual int32_t Initialize(const TableParameter &config,
                             const FsClientParameter &fs_config);
  virtual int32_t Initialize(const GraphParameter &config);
  void init_worker_poll(int gpu_num);
  int32_t Load(const std::string &path, const std::string &param);
  int32_t load_node_and_edge_file(std::string etype2files,
                                  std::string ntype2files,
                                  std::string graph_data_local_path,
                                  int part_num,
                                  bool reverse,
                                  const std::vector<bool> &is_reverse_edge_map,
                                  bool use_weight);
  int32_t parse_edge_and_load(std::string etype2files,
                              std::string graph_data_local_path,
                              int part_num,
                              bool reverse,
                              const std::vector<bool> &is_reverse_edge_map,
                              bool use_weight);
  int32_t parse_node_and_load(std::string ntype2files,
                              std::string graph_data_local_path,
                              int part_num,
                              bool load_slot = true);
  std::string get_inverse_etype(std::string &etype);  // NOLINT
  int32_t parse_type_to_typepath(
      std::string &type2files,  // NOLINT
      std::string graph_data_local_path,
      std::vector<std::string> &res_type,                            // NOLINT
      std::unordered_map<std::string, std::string> &res_type2path);  // NOLINT
  std::pair<uint64_t, uint64_t> load_edges(const std::string &path,
                                           bool reverse,
                                           const std::string &edge_type,
                                           bool use_weight = false);
  int get_all_id(GraphTableType table_type,
                 int slice_num,
                 std::vector<std::vector<uint64_t>> *output);
  int get_all_neighbor_id(GraphTableType table_type,
                          int slice_num,
                          std::vector<std::vector<uint64_t>> *output);
  int get_all_id(GraphTableType table_type,
                 int idx,
                 int slice_num,
                 std::vector<std::vector<uint64_t>> *output);
  int get_all_neighbor_id(GraphTableType table_type,
                          int id,
                          int slice_num,
                          std::vector<std::vector<uint64_t>> *output);
  int get_all_feature_ids(GraphTableType table_type,
                          int idx,
                          int slice_num,
                          std::vector<std::vector<uint64_t>> *output);
  int get_node_embedding_ids(int slice_num,
                             std::vector<std::vector<uint64_t>> *output);
  int32_t load_nodes(const std::string &path,
                     std::string node_type = std::string(),
                     bool load_slot = true);
  std::pair<uint64_t, uint64_t> parse_edge_file(const std::string &path,
                                                int idx,
                                                bool reverse,
                                                bool use_weight);
  std::pair<uint64_t, uint64_t> parse_node_file(const std::string &path,
                                                const std::string &node_type,
                                                int idx,
                                                bool load_slot = true);
  std::pair<uint64_t, uint64_t> parse_node_file_parallel(
      const std::string &path, bool load_slot = true);
  int32_t add_graph_node(int idx,
                         std::vector<uint64_t> &id_list,      // NOLINT
                         std::vector<bool> &is_weight_list);  // NOLINT

  int32_t remove_graph_node(int idx, std::vector<uint64_t> &id_list);  // NOLINT

  int32_t get_server_index_by_id(uint64_t id);
  Node *find_node(GraphTableType table_type, int idx, uint64_t id);
  Node *find_node(GraphTableType table_type, uint64_t id);
  // query all ids rank
  void query_all_ids_rank(const size_t &total,
                          const uint64_t *ids,
                          uint32_t *ranks);

  virtual int32_t Pull(TableContext &context UNUSED) { return 0; }  // NOLINT
  virtual int32_t Push(TableContext &context UNUSED) { return 0; }  // NOLINT

  virtual int32_t clear_nodes(GraphTableType table_type, int idx);
  virtual void Clear() {}
  virtual int32_t Flush() { return 0; }
  virtual int32_t Shrink(const std::string &param UNUSED) { return 0; }
  // 指定保存路径
  virtual int32_t Save(const std::string &path UNUSED,
                       const std::string &converter UNUSED) {
    return 0;
  }
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  virtual int32_t Save_v2(const std::string &path,
                          const std::string &converter) {
    return 0;
  }
#endif
  virtual int32_t InitializeShard() { return 0; }
  virtual int32_t SetShard(size_t shard_idx, size_t server_num) {
    _shard_idx = shard_idx;
    /*
    _shard_num is not used in graph_table, this following operation is for the
    purpose of
    being compatible with base class table.
    */
    _shard_num = server_num;
    this->server_num = server_num;
    return 0;
  }

  virtual uint32_t get_thread_pool_index_by_shard_index(uint64_t shard_index);
  virtual uint32_t get_thread_pool_index(uint64_t node_id);
  virtual int parse_feature(int idx,
                            const char *feat_str,
                            size_t len,
                            FeatureNode *node);

  virtual int32_t get_node_feat(
      int idx,
      const std::vector<uint64_t> &node_ids,
      const std::vector<std::string> &feature_names,
      std::vector<std::vector<std::string>> &res);  // NOLINT

  virtual int32_t set_node_feat(
      int idx,
      const std::vector<uint64_t> &node_ids,              // NOLINT
      const std::vector<std::string> &feature_names,      // NOLINT
      const std::vector<std::vector<std::string>> &res);  // NOLINT

  size_t get_server_num() { return server_num; }
  void clear_graph();
  void clear_graph(int idx);
  void clear_edge_shard();
  void clear_feature_shard();
  void clear_node_shard();
  void feature_shrink_to_fit();
  void merge_feature_shard();
  void release_graph();
  void release_graph_edge();
  void release_graph_node();
  virtual int32_t make_neighbor_sample_cache(size_t size_limit, size_t ttl) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (use_cache == false) {
        scaled_lru.reset(new ScaledLRU<SampleKey, SampleResult>(
            task_pool_size_, size_limit, ttl));
        use_cache = true;
      }
    }
    return 0;
  }
  virtual void load_node_weight(int type_id, int idx, std::string path);
#ifdef PADDLE_WITH_HETERPS
  virtual void make_partitions(int idx, int64_t gb_size, int device_len);
  virtual void export_partition_files(int idx, std::string file_path);
  virtual char *random_sample_neighbor_from_ssd(
      int idx,
      uint64_t id,
      int sample_size,
      const std::shared_ptr<std::mt19937_64> rng,
      int &actual_size);  // NOLINT
  virtual int32_t add_node_to_ssd(
      int type_id, int idx, uint64_t src_id, char *data, int len);
  virtual ::paddle::framework::GpuPsCommGraph make_gpu_ps_graph(
      int idx, const std::vector<uint64_t> &ids);
  virtual ::paddle::framework::GpuPsCommGraphFea make_gpu_ps_graph_fea(
      int gpu_id, std::vector<uint64_t> &node_ids, int slot_num);  // NOLINT
  virtual paddle::framework::GpuPsCommGraphFloatFea make_gpu_ps_graph_float_fea(
      int gpu_id,
      std::vector<uint64_t> &node_ids,  // NOLINT
      int float_slot_num);
  virtual paddle::framework::GpuPsCommRankFea make_gpu_ps_rank_fea(int gpu_id);
  int32_t Load_to_ssd(const std::string &path, const std::string &param);
  int64_t load_graph_to_memory_from_ssd(int idx,
                                        std::vector<uint64_t> &ids);  // NOLINT
  int32_t make_complementary_graph(int idx, int64_t byte_size);
  int32_t dump_edges_to_ssd(int idx);
  int32_t get_partition_num(int idx) { return partitions[idx].size(); }
  std::vector<int> slot_feature_num_map() const {
    return slot_feature_num_map_;
  }
  std::vector<uint64_t> get_partition(size_t idx, size_t index) {
    if (idx >= partitions.size() || index >= partitions[idx].size())
      return std::vector<uint64_t>();
    return partitions[idx][index];
  }
  int32_t load_edges_to_ssd(const std::string &path,
                            bool reverse_edge,
                            const std::string &edge_type);
  int32_t load_next_partition(int idx);
  void set_search_level(int search_level) { this->search_level = search_level; }

  void graph_partition(bool is_edge);
  void dbh_graph_edge_partition();
  void dbh_graph_feature_partition();
  void fennel_graph_edge_partition();
  void filter_graph_edge_nodes();
  void fennel_graph_feature_partition();

  int search_level;
  int64_t total_memory_cost;
  std::vector<std::vector<std::vector<uint64_t>>> partitions;
  int next_partition;
#endif
  virtual int32_t add_comm_edge(int idx, uint64_t src_id, uint64_t dst_id);
  virtual int32_t build_sampler(int idx, std::string sample_type = "random");
  void set_slot_feature_separator(const std::string &ch);
  void set_feature_separator(const std::string &ch);

  void build_graph_total_keys();
  void build_graph_type_keys();
  void calc_edge_type_limit();
  void build_node_iter_type_keys();
  bool is_key_for_self_rank(const uint64_t &id);
  int partition_key_for_rank(const uint64_t &key);
  void fix_feature_node_shards(bool load_slot);
  void stat_graph_edge_info(int type);
  std::string node_types_idx_to_node_type_str(int node_types_idx);
  std::string index_to_node_type_str(int index);

  std::vector<uint64_t> graph_total_keys_;
  std::vector<std::vector<uint64_t>> graph_type_keys_;
  std::unordered_map<int, int> type_to_index_;
  std::unordered_map<int, int> index_to_type_;
  std::vector<std::string> node_types_;
  robin_hood::unordered_set<uint64_t> unique_all_edge_keys_;
  // node 2 rank
  GraphNodeRank edge_node_rank_;
  std::unordered_map<int, int> type_to_neighbor_limit_;

  std::vector<std::vector<GraphShard *>> edge_shards, feature_shards,
      node_shards;
  size_t shard_start, shard_end, server_num, shard_num_per_server, shard_num;
  int task_pool_size_ = 64;
  int load_thread_num_ = 160;
  std::vector<std::vector<std::vector<uint64_t>>> edge_shards_keys_;

  const int random_sample_nodes_ranges = 3;

  std::vector<std::vector<std::unordered_map<uint64_t, double>>> node_weight;
  std::vector<std::vector<std::string>> feat_name;
  std::vector<std::vector<std::string>> feat_dtype;
  std::vector<std::vector<int32_t>> feat_shape;
  std::vector<std::vector<std::string>> float_feat_name;
  std::vector<std::vector<std::string>> float_feat_dtype;
  std::vector<std::vector<int32_t>> float_feat_shape;
  // int slot_fea_num_{-1};
  // int float_fea_num_{-1};
  std::vector<std::unordered_map<std::string, int32_t>> feat_id_map;
  std::vector<std::unordered_map<std::string, int32_t>> float_feat_id_map;
  std::unordered_map<std::string, int> node_type_str_to_node_types_idx,
      edge_to_id;
  std::vector<std::string> id_to_feature, id_to_edge;
  std::string table_name;
  std::string table_type;
  std::vector<std::string> edge_type_size;
  std::vector<std::vector<int>> nodeid_to_edgeids_;

  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
  std::vector<std::shared_ptr<::ThreadPool>> _cpu_worker_pool;
  std::vector<std::shared_ptr<std::mt19937_64>> _shards_task_rng_pool;
  std::shared_ptr<::ThreadPool> load_node_edge_task_pool;
  std::shared_ptr<ScaledLRU<SampleKey, SampleResult>> scaled_lru;
  std::unordered_set<uint64_t> extra_nodes;
  std::unordered_map<uint64_t, size_t> extra_nodes_to_thread_index;
  bool use_cache, use_duplicate_nodes;
  int cache_size_limit;
  int cache_ttl;
  mutable std::mutex mutex_;
  bool build_sampler_on_cpu;
  bool is_load_reverse_edge = false;
  std::shared_ptr<pthread_rwlock_t> rw_lock;
#ifdef PADDLE_WITH_HETERPS
  // paddle::framework::GpuPsGraphTable gpu_graph_table;
  ::paddle::distributed::RocksDBHandler *_db;
  // std::shared_ptr<::ThreadPool> graph_sample_pool;
  // std::shared_ptr<GraphSampler> graph_sampler;
  // REGISTER_GRAPH_FRIEND_CLASS(2, CompleteGraphSampler, BasicBfsGraphSampler)
#endif
  std::string slot_feature_separator_ = std::string(" ");
  std::string feature_separator_ = std::string(" ");
  std::vector<int> slot_feature_num_map_;
  bool is_parse_node_fail_ = false;
  int node_num_ = 1;
  int node_id_ = 0;
  bool is_weighted_ = false;
};
}  // namespace distributed

};  // namespace paddle

namespace std {

template <>
struct hash<::paddle::distributed::SampleKey> {
  size_t operator()(const ::paddle::distributed::SampleKey &s) const {
    return s.idx ^ s.node_key ^ s.sample_size;
  }
};
}  // namespace std
