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

#pragma once

#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/ps/table/depends/rocksdb_wrapper.h"
#include "paddle/fluid/distributed/ps/table/memory_sparse_table.h"

#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#endif

namespace paddle {
namespace distributed {

class MemRegion {
 public:
  MemRegion() {
    _cap = 2 * 1024 * 1024;
    _buf = reinterpret_cast<char*>(malloc(_cap));
    _cur = 0;
    _file_idx = -1;
  }
  virtual ~MemRegion() { free(_buf); }
  bool buff_remain(int len) {
    if (_cap - _cur < len) {
      return false;
    } else {
      return true;
    }
  }
  char* acquire(int len) {
    if (_cap - _cur < len) {
      return nullptr;
    } else {
      char* ret = _buf + _cur;
      _cur += len;
      return ret;
    }
  }
  void reset() {
    _cur = 0;
    _file_idx = -1;
  }
  int _cap;
  int _cur;
  int _file_idx;
  char* _buf;
};

class SSDSparseTable : public MemorySparseTable {
 public:
  typedef SparseTableShard<uint64_t, FixedFeatureValue> shard_type;
  SSDSparseTable() {}
  virtual ~SSDSparseTable() {}

  int32_t Initialize() override;
  int32_t InitializeShard() override;

  // exchange data
  int32_t UpdateTable();

  int32_t Pull(TableContext& context) override;

  int32_t Push(TableContext& context) override;

  int32_t PullSparse(float* pull_values, const uint64_t* keys, size_t num);
  int32_t PullSparsePtr(int shard_id,
                        char** pull_values,
                        const uint64_t* keys,
                        size_t num,
                        uint16_t pass_id);
  int32_t PushSparse(const uint64_t* keys, const float* values, size_t num);
  int32_t PushSparse(const uint64_t* keys, const float** values, size_t num);

  int32_t Flush() override { return 0; }
  int32_t Shrink(const std::string& param) override;
  void Clear() override {
    for (int i = 0; i < _real_local_shard_num; ++i) {
      _local_shards[i].clear();
    }
  }

  int32_t Save(const std::string& path, const std::string& param) override;
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  int32_t Save_v2(const std::string& path, const std::string& param) override;
#endif
  int32_t SaveWithString(const std::string& path, const std::string& param);
  int32_t SaveWithStringMultiOutput(const std::string& path,
                                    const std::string& param);
  int32_t SaveWithStringMultiOutput_v2(const std::string& path,
                                       const std::string& param);
  int32_t SaveWithBinary(const std::string& path, const std::string& param);
  int32_t SaveWithBinary_v2(const std::string& path, const std::string& param);
  int32_t SaveCache(
      const std::string& path,
      const std::string& param,
      paddle::framework::Channel<std::pair<uint64_t, std::string>>&
          shuffled_channel) override;
  double GetCacheThreshold() override { return _local_show_threshold; }
  int64_t CacheShuffle(
      const std::string& path,
      const std::string& param,
      double cache_threshold,
      std::function<std::future<int32_t>(
          int msg_type, int to_pserver_id, std::string& msg)> send_msg_func,
      paddle::framework::Channel<std::pair<uint64_t, std::string>>&
          shuffled_channel,
      const std::vector<Table*>& table_ptrs) override;
  // 加载path目录下数据
  int32_t Load(const std::string& path, const std::string& param) override;
  int32_t LoadWithString(size_t file_start_idx,
                         size_t end_idx,
                         const std::vector<std::string>& file_list,
                         const std::string& param);
  int32_t LoadWithBinary(const std::string& path, int param);
  int64_t LocalSize();

  std::pair<int64_t, int64_t> PrintTableStat() override;

  int32_t CacheTable(uint16_t pass_id) override;

  void SetDayId(int day_id) override;

 private:
  RocksDBHandler* _db;
  int64_t _cache_tk_size;
  double _local_show_threshold{0.0};
  std::vector<paddle::framework::Channel<std::string>> _fs_channel;
  std::mutex _table_mutex;
  int _day_id = 0;
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  paddle::framework::AfsWrapper _afs_wrapper;  // afs api wrapper
#endif
  bool _use_afs_api = false;
};

}  // namespace distributed
}  // namespace paddle
