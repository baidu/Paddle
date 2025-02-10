// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <assert.h>

#include <atomic>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <utility>

#include "paddle/common/macros.h"
#include "paddle/fluid/distributed/common/afs_wrapper.h"
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/depends/sparse_utils.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace distributed {

enum ValueType { Sparse = 0, Dense = 1 };

struct TablePullContext {
  const uint64_t *keys = nullptr;
  PullSparseValue pull_value;
  float *values = nullptr;
  char **ptr_values = nullptr;
  std::vector<uint64_t> *geo_pull_keys = nullptr;  // for GEO
  std::vector<float> *geo_pull_values = nullptr;   // for GEO
};

struct TablePushContext {
  const uint64_t *keys = nullptr;
  const float *values = nullptr;
  const float **ptr_values = nullptr;
  const int64_t *push_steps = nullptr;  // for global step
  bool is_param = false;  // true: push param, false: push gradient
};

struct TableContext {
  ValueType value_type;
  TablePullContext pull_context;
  TablePushContext push_context;
  size_t num;
  bool use_ptr = false;
  uint32_t trainer_id;  // for GEO and global step
  int shard_id;         // for gpups
  uint16_t pass_id;     // for gpups ssd
};

class Table {
 public:
  Table() {}
  virtual ~Table() {}
  virtual int32_t Initialize(const TableParameter &config,
                             const FsClientParameter &fs_config);

  virtual int32_t Pull(TableContext &context) = 0;  // NOLINT
  virtual int32_t Push(TableContext &context) = 0;  // NOLINT

  // only for barrier
  virtual int32_t Barrier(const uint32_t trainer_id UNUSED,
                          const std::string barrier_type UNUSED) {
    return 0;
  }

  // only for barrier table
  virtual int32_t SetTableMap(
      std::unordered_map<uint32_t, std::shared_ptr<Table>> *table_map UNUSED) {
    return 0;
  }

  // only for tensor table
  virtual int32_t SetProgramEnv(
      framework::Scope *scope UNUSED,
      phi::Place place UNUSED,
      const std::vector<framework::ProgramDesc> *sub_program UNUSED) {
    return 0;
  }

  virtual int32_t SetGlobalLR(float *lr) {
    _global_lr = lr;
    return 0;
  }

  virtual int32_t Pour() { return 0; }

  virtual void Clear() = 0;
  virtual int32_t Flush() = 0;
  virtual int32_t Shrink(const std::string &param) = 0;

  // 指定加载路径
  virtual int32_t Load(const std::string &path,
                       const std::string &converter) = 0;
  // 指定保存路径
  virtual int32_t Save(const std::string &path,
                       const std::string &converter) = 0;

#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  // pglbox支持将非9008 slot的feature额外保存一份，实际支持用户可配置过滤slot
  virtual int32_t Save_v2(const std::string &path,
                          const std::string &converter) = 0;
#endif

  // for cache
  virtual int32_t SaveCache(
      const std::string &path UNUSED,
      const std::string &param UNUSED,
      ::paddle::framework::Channel<std::pair<uint64_t, std::string>>
          &shuffled_channel UNUSED) {
    return 0;
  }

  virtual int64_t CacheShuffle(
      const std::string &path UNUSED,
      const std::string &param UNUSED,
      double cache_threshold UNUSED,
      std::function<std::future<int32_t>(
          int msg_type, int to_pserver_id, std::string &msg)>  // NOLINT
          send_msg_func UNUSED,
      ::paddle::framework::Channel<std::pair<uint64_t, std::string>>
          &shuffled_channel UNUSED,
      const std::vector<Table *> &table_ptrs UNUSED) {
    return 0;
  }

  virtual double GetCacheThreshold() { return 0.0; }

  virtual int32_t SetShard(size_t shard_idx, size_t shard_num) {
    _shard_idx = shard_idx;
    _shard_num = shard_num;
    return InitializeShard();
  }

  inline std::shared_ptr<ValueAccessor> GetValueAccessor() {
    return _value_accessor;
  }

  virtual void *GetShard(size_t shard_idx) = 0;
  virtual std::pair<int64_t, int64_t> PrintTableStat() { return {0, 0}; }
  virtual int32_t CacheTable(uint16_t pass_id UNUSED) { return 0; }

  // for patch model
  virtual void Revert() {}
  virtual void CheckSavePrePatchDone() {}

  virtual void SetDayId(int day_id) {}

 protected:
  virtual int32_t Initialize() = 0;
  virtual int32_t InitializeAccessor();
  virtual int32_t InitializeShard() = 0;
  virtual std::string TableDir(const std::string &model_dir) {
    return ::paddle::string::format_string(
        "%s/%03d/", model_dir.c_str(), _config.table_id());
  }

  size_t _shard_idx;  // table 分片编号
  size_t _shard_num;  // table 分片总数
  TableParameter _config;
  float *_global_lr = nullptr;
  std::shared_ptr<ValueAccessor> _value_accessor;
  AfsClient _afs_client;
  std::string _fs_name;
  std::string _fs_user;
  std::string _pass_wd;
};
REGISTER_PSCORE_REGISTERER(Table);

class TableManager {
 public:
  static TableManager &Instance() {
    static TableManager manager;
    return manager;
  }
  int32_t Initialize();

 private:
  TableManager() {}
  ~TableManager() {}
};

}  // namespace distributed
}  // namespace paddle
