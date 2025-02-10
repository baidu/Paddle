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

#include <glog/logging.h>
#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/iostats_context.h>
#include <rocksdb/options.h>
#include <rocksdb/perf_context.h>
#include <rocksdb/perf_level.h>
#include <rocksdb/slice.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include <rocksdb/write_batch.h>

#include <iostream>
#include <string>

namespace paddle {
namespace distributed {

class Uint64Comparator : public rocksdb::Comparator {
  int Compare(const rocksdb::Slice& a, const rocksdb::Slice& b) const {
    uint64_t A = *(reinterpret_cast<const uint64_t*>(a.data()));
    uint64_t B = *(reinterpret_cast<const uint64_t*>(b.data()));
    if (A < B) {
      return -1;
    }
    if (A > B) {
      return 1;
    }
    return 0;
  }
  const char* Name() const { return "Uint64Comparator"; }
  void FindShortestSeparator(std::string*, const rocksdb::Slice&) const {}
  void FindShortSuccessor(std::string*) const {}
};

class RocksDBItem {
 public:
  RocksDBItem() {}
  ~RocksDBItem() {}
  void reset() {
    batch_keys.clear();
    batch_index.clear();
    batch_values.clear();
    status.clear();
  }
  std::vector<rocksdb::Slice> batch_keys;
  std::vector<int> batch_index;
  std::vector<rocksdb::PinnableSlice> batch_values;
  std::vector<rocksdb::Status> status;
};

class RocksDBCtx {
 public:
  RocksDBCtx() {
    items[0].reset();
    items[1].reset();
    cur_index = 0;
  }
  ~RocksDBCtx() {}
  RocksDBItem* switch_item() {
    cur_index = (cur_index + 1) % 2;
    return &items[cur_index];
  }
  RocksDBItem items[2];
  int cur_index;
};

class RocksDBHandler {
 public:
  RocksDBHandler() {}
  ~RocksDBHandler() {}

  static RocksDBHandler* GetInstance() {
    static RocksDBHandler handler;
    return &handler;
  }

  int initialize(const std::string& db_path, const int colnum) {
    VLOG(0) << "db path: " << db_path << " colnum: " << colnum;
    _dbs.resize(colnum);
    for (int i = 0; i < colnum; i++) {
      rocksdb::Options options;
      options.comparator = &_comparator;
      rocksdb::BlockBasedTableOptions bbto;
      // options.memtable_factory.reset(rocksdb::NewHashSkipListRepFactory(65536));
      // options.prefix_extractor.reset(rocksdb::NewFixedPrefixTransform(2));
      bbto.format_version = 5;
      bbto.use_delta_encoding = false;
      bbto.block_size = 4 * 1024;
      bbto.block_restart_interval = 6;
      bbto.block_cache = rocksdb::NewLRUCache(64 * 1024 * 1024);
      // bbto.block_cache_compressed = rocksdb::NewLRUCache(64 * 1024 * 1024);
      bbto.cache_index_and_filter_blocks = false;
      bbto.filter_policy.reset(rocksdb::NewBloomFilterPolicy(15, false));
      bbto.whole_key_filtering = true;
      options.statistics = rocksdb::CreateDBStatistics();
      options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(bbto));

      // options.IncreaseParallelism();
      options.OptimizeLevelStyleCompaction();
      options.keep_log_file_num = 100;
      // options.db_log_dir = "./log/rocksdb";
      options.max_log_file_size = 50 * 1024 * 1024;  // 50MB
      // options.threads = 8;
      options.create_if_missing = true;
      options.use_direct_reads = true;
      options.max_background_flushes = 37;
      options.max_background_compactions = 64;
      options.base_background_compactions = 10;
      options.write_buffer_size = 256 * 1024 * 1024;  // 256MB
      options.max_write_buffer_number = 8;
      options.max_bytes_for_level_base =
          options.max_write_buffer_number * options.write_buffer_size;
      options.min_write_buffer_number_to_merge = 1;
      options.target_file_size_base = 1024 * 1024 * 1024;  // 1024MB
      // options.verify_checksums_in_compaction = false;
      // options.disable_auto_compactions = true;
      options.memtable_prefix_bloom_size_ratio = 0.02;
      options.num_levels = 4;
      options.max_open_files = -1;

      options.compression = rocksdb::kNoCompression;
      // options.compaction_options_fifo = rocksdb::CompactionOptionsFIFO();
      // options.compaction_style =
      // rocksdb::CompactionStyle::kCompactionStyleFIFO;
      options.level0_file_num_compaction_trigger = 5;
      options.level0_slowdown_writes_trigger =
          1.8 * options.level0_file_num_compaction_trigger;
      options.level0_stop_writes_trigger =
          3.6 * options.level0_file_num_compaction_trigger;

      std::string shard_path = db_path + "_" + std::to_string(i);
      if (!shard_path.empty()) {
        std::string rm_cmd = "rm -rf " + shard_path;
        system(rm_cmd.c_str());
      }

      rocksdb::Status s = rocksdb::DB::Open(options, shard_path, &_dbs[i]);
      assert(s.ok());
    }
    VLOG(0) << "DB initialize success, colnum:" << colnum;
    return 0;
  }

  int put(
      int id, const char* key, int key_len, const char* value, int value_len) {
    rocksdb::WriteOptions options;
    options.disableWAL = true;
    rocksdb::Status s = _dbs[id]->Put(options,
                                      rocksdb::Slice(key, key_len),
                                      rocksdb::Slice(value, value_len));
    assert(s.ok());
    return 0;
  }

  int put_batch(int id,
                std::vector<std::pair<char*, int>>& ssd_keys,    // NOLINT
                std::vector<std::pair<char*, int>>& ssd_values,  // NOLINT
                int n) {
    rocksdb::WriteOptions options;
    options.disableWAL = true;
    rocksdb::WriteBatch batch(n * 128);
    for (int i = 0; i < n; i++) {
      batch.Put(rocksdb::Slice(ssd_keys[i].first, ssd_keys[i].second),
                rocksdb::Slice(ssd_values[i].first, ssd_values[i].second));
    }
    rocksdb::Status s = _dbs[id]->Write(options, &batch);
    assert(s.ok());
    return 0;
  }

  int get(int id, const char* key, int key_len, std::string& value) {  // NOLINT
    rocksdb::Status s = _dbs[id]->Get(
        rocksdb::ReadOptions(), rocksdb::Slice(key, key_len), &value);
    if (s.IsNotFound()) {
      return 1;
    }
    assert(s.ok());
    return 0;
  }

  void multi_get(int id,
                 const size_t num_keys,
                 const rocksdb::Slice* keys,
                 rocksdb::PinnableSlice* values,
                 rocksdb::Status* status,
                 const bool sorted_input = true) {
    rocksdb::ColumnFamilyHandle* handle = _dbs[id]->DefaultColumnFamily();
    auto read_opt = rocksdb::ReadOptions();
    read_opt.fill_cache = false;
    _dbs[id]->MultiGet(
        read_opt, handle, num_keys, keys, values, status, sorted_input);
  }

  int del_data(int id, const char* key, int key_len) {
    rocksdb::WriteOptions options;
    options.disableWAL = true;
    rocksdb::Status s = _dbs[id]->Delete(options, rocksdb::Slice(key, key_len));
    assert(s.ok());
    return 0;
  }

  int flush(int id) {
    rocksdb::Status s = _dbs[id]->Flush(rocksdb::FlushOptions());
    assert(s.ok());
    return 0;
  }

  rocksdb::Iterator* get_iterator(int id) {
    return _dbs[id]->NewIterator(rocksdb::ReadOptions());
  }

  int get_estimate_key_num(uint64_t& num_keys) {  // NOLINT
    num_keys = 0;
    for (size_t i = 0; i < _dbs.size(); i++) {
      uint64_t cur_keys = 0;
      _dbs[i]->GetAggregatedIntProperty("rocksdb.estimate-num-keys", &cur_keys);
      num_keys += cur_keys;
    }
    return 0;
  }

  Uint64Comparator* get_comparator() { return &_comparator; }

  int ingest_external_file(int id,
                           const std::vector<std::string>& sst_filelist) {
    rocksdb::IngestExternalFileOptions ifo;
    ifo.move_files = true;
    rocksdb::Status s = _dbs[id]->IngestExternalFile(sst_filelist, ifo);
    assert(s.ok());
    return 0;
  }

 private:
  std::vector<rocksdb::ColumnFamilyHandle*> _handles;
  // rocksdb::DB* _db;
  std::vector<rocksdb::DB*> _dbs;
  Uint64Comparator _comparator;
};
}  // namespace distributed
}  // namespace paddle
