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

#include <omp.h>
#include <sstream>

#include "glog/logging.h"
#include "paddle/fluid/distributed/common/cost_timer.h"
#include "paddle/fluid/distributed/common/local_random.h"
#include "paddle/fluid/distributed/common/topk_calculator.h"
#include "paddle/fluid/distributed/ps/table/memory_sparse_table.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/framework/io/fs.h"

// #include "boost/lexical_cast.hpp"
#include "paddle/fluid/platform/enforce.h"

PD_DEFINE_bool(pserver_print_missed_key_num_every_push,
               false,
               "pserver_print_missed_key_num_every_push");
PD_DEFINE_bool(pserver_create_value_when_push,
               true,
               "pserver create value when push");
PD_DEFINE_bool(pserver_enable_create_feasign_randomly,
               false,
               "pserver_enable_create_feasign_randomly");
PD_DEFINE_int32(pserver_table_save_max_retry,
                3,
                "pserver_table_save_max_retry");

namespace paddle::distributed {

int32_t MemorySparseTable::Initialize() {
  auto &profiler = CostProfiler::instance();
  profiler.register_profiler("pserver_sparse_update_all");
  profiler.register_profiler("pserver_sparse_select_all");
  InitializeValue();
  _shards_task_pool.resize(_task_pool_size);
  for (auto &shards_task : _shards_task_pool) {
    shards_task.reset(new ::ThreadPool(1));
  }
  VLOG(0) << "initialize MemorySparseTable succ";
  return 0;
}

int32_t MemorySparseTable::InitializeValue() {
  _sparse_table_shard_num = static_cast<int>(_config.shard_num());
  _avg_local_shard_num =
      sparse_local_shard_num(_sparse_table_shard_num, _shard_num);
  _real_local_shard_num = _avg_local_shard_num;
  if (static_cast<int>(_real_local_shard_num * (_shard_idx + 1)) >
      _sparse_table_shard_num) {
    _real_local_shard_num =
        _sparse_table_shard_num - _real_local_shard_num * _shard_idx;
    _real_local_shard_num =
        _real_local_shard_num < 0 ? 0 : _real_local_shard_num;
  }
#ifdef PADDLE_WITH_HETERPS
  _task_pool_size = _sparse_table_shard_num;
#endif
  _use_gpu_graph = _config.use_gpu_graph();
  VLOG(1) << "memory sparse table _avg_local_shard_num: "
          << _avg_local_shard_num
          << " _real_local_shard_num: " << _real_local_shard_num
          << " _task_pool_size:" << _task_pool_size
          << " _use_gpu_graph:" << _use_gpu_graph;

  _local_shards.reset(new shard_type[_real_local_shard_num]);

  if (_config.enable_revert()) {
    // calculate merged shard number based on config param;
    _shard_merge_rate = _config.has_shard_merge_rate()
                            ? _config.shard_merge_rate()
                            : _shard_merge_rate;
    _m_avg_local_shard_num =
        static_cast<int>(std::ceil(_avg_local_shard_num * _shard_merge_rate));
    PADDLE_ENFORCE_LE(
        _m_avg_local_shard_num,
        _avg_local_shard_num,
        common::errors::InvalidArgument(
            "The calculated '_m_avg_local_shard_num' (%d) must be less than or "
            "equal to '_avg_local_shard_num' (%d).",
            _m_avg_local_shard_num,
            _avg_local_shard_num));

    _m_real_local_shard_num =
        static_cast<int>(std::ceil(_real_local_shard_num * _shard_merge_rate));
    PADDLE_ENFORCE_LE(
        _m_real_local_shard_num,
        _real_local_shard_num,
        common::errors::InvalidArgument(
            "The calculated '_m_real_local_shard_num' (%d) must be less than "
            "or equal to '_real_local_shard_num' (%d).",
            _m_real_local_shard_num,
            _real_local_shard_num));

    uint32_t avg_shard_server_num =
        _sparse_table_shard_num / _avg_local_shard_num;
    uint32_t last_server_shard_num =
        _sparse_table_shard_num - avg_shard_server_num * _avg_local_shard_num;
    _m_sparse_table_shard_num =
        avg_shard_server_num * _m_avg_local_shard_num +
        std::ceil(last_server_shard_num * _shard_merge_rate);
    LOG(INFO) << "merged shard info: [" << _m_sparse_table_shard_num << "|"
              << _m_avg_local_shard_num << "|" << _m_real_local_shard_num
              << "]";
    _local_shards_new.reset(new shard_type[_real_local_shard_num]);  // NOLINT
  }
  return 0;
}

int32_t MemorySparseTable::Load(const std::string &path,
                                const std::string &param) {
  std::string table_path = TableDir(path);
  auto file_list = _afs_client.list(table_path);

  std::sort(file_list.begin(), file_list.end());
  for (auto file : file_list) {
    VLOG(1) << "MemorySparseTable::Load() file list: " << file;
  }

  int load_param = atoi(param.c_str());
  size_t expect_shard_num = _sparse_table_shard_num;
  if (file_list.size() != expect_shard_num) {
    LOG(WARNING) << "MemorySparseTable file_size:" << file_list.size()
                 << " not equal to expect_shard_num:" << expect_shard_num;
    return -1;
  }
  if (file_list.empty()) {
    LOG(WARNING) << "MemorySparseTable load file is empty, path:" << path;
    return -1;
  }

  if (load_param == 5) {
    return LoadPatch(file_list, load_param);
  }

  size_t file_start_idx = _shard_idx * _avg_local_shard_num;

  if (file_start_idx >= file_list.size()) {
    return 0;
  }

  size_t feature_value_size =
      _value_accessor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_size =
      _value_accessor->GetAccessorInfo().mf_size / sizeof(float);

#ifdef PADDLE_WITH_HETERPS
  int thread_num = _real_local_shard_num;
#else
  int thread_num = _real_local_shard_num < 15 ? _real_local_shard_num : 15;
#endif

  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < _real_local_shard_num; ++i) {
    uint64_t mem_count = 0;
    uint64_t mem_mf_count = 0;

    FsChannelConfig channel_config = {};
    channel_config.path = file_list[file_start_idx + i];
    VLOG(1) << "MemorySparseTable::load begin load " << channel_config.path
            << " into local shard " << i;
    channel_config.converter = _value_accessor->Converter(load_param).converter;
    channel_config.deconverter =
        _value_accessor->Converter(load_param).deconverter;

    bool is_read_failed = false;
    int retry_num = 0;
    int err_no = 0;
    do {
      is_read_failed = false;
      err_no = 0;
      std::string line_data;
      auto read_channel = _afs_client.open_r(channel_config, 0, &err_no);
      char *end = nullptr;
      auto &shard = _local_shards[i];
      try {
        while (read_channel->read_line(line_data) == 0 &&
               line_data.size() > 1) {
          uint64_t key = std::strtoul(line_data.data(), &end, 10);
          auto &value = shard[key];
          value.resize(feature_value_size);
          int parse_size =
              _value_accessor->ParseFromString(++end, value.data());
          mem_count++;
          value.resize(parse_size);
          if (parse_size >
              static_cast<int>(feature_value_size - mf_value_size)) {
            mem_mf_count++;
          }
        }
        read_channel->close();
        if (err_no == -1) {
          ++retry_num;
          is_read_failed = true;
          LOG(ERROR)
              << "MemorySparseTable load failed after read, retry it! path:"
              << channel_config.path << " , retry_num=" << retry_num;
        }
      } catch (...) {
        ++retry_num;
        is_read_failed = true;
        LOG(ERROR) << "MemorySparseTable load failed, retry it! path:"
                   << channel_config.path << " , retry_num=" << retry_num;
      }
      if (retry_num > FLAGS_pserver_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable load failed reach max limit!";
        exit(-1);
      }
    } while (is_read_failed);
    VLOG(0) << "Table>> load done. ALL[" << mem_count << "] MEM[" << mem_count
            << "] MEM_MF[" << mem_mf_count << "]";
  }
  LOG(INFO) << "MemorySparseTable load success, path from "
            << file_list[file_start_idx] << " to "
            << file_list[file_start_idx + _real_local_shard_num - 1];
  return 0;
}

int32_t MemorySparseTable::LoadPatch(const std::vector<std::string> &file_list,
                                     int load_param) {
  if (!_config.enable_revert()) {
    LOG(INFO) << "MemorySparseTable should be enabled revert.";
    return 0;
  }
  // 聚合分片数据索引
  int start_idx = _shard_idx * _m_avg_local_shard_num;
  int end_idx = start_idx + _m_real_local_shard_num;
  // 原始分片数据索引
  int o_start_idx = _shard_idx * _avg_local_shard_num;
  int o_end_idx = o_start_idx + _real_local_shard_num;

  if (start_idx >= static_cast<int>(file_list.size())) {
    return 0;
  }
  size_t feature_value_size =
      _value_accessor->GetAccessorInfo().size / sizeof(float);
  end_idx =
      end_idx < _m_sparse_table_shard_num ? end_idx : _m_sparse_table_shard_num;
  int thread_num = (end_idx - start_idx) < 15 ? (end_idx - start_idx) : 15;

  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (int i = start_idx; i < end_idx; ++i) {
    FsChannelConfig channel_config = {};
    channel_config.path = file_list[i];
    channel_config.converter = _value_accessor->Converter(load_param).converter;
    channel_config.deconverter =
        _value_accessor->Converter(load_param).deconverter;

    bool is_read_failed = false;
    int retry_num = 0;
    int err_no = 0;
    do {
      is_read_failed = false;
      err_no = 0;
      std::string line_data;
      auto read_channel = _afs_client.open_r(channel_config, 0, &err_no);
      char *end = nullptr;
      int m_local_shard_id = i % _m_avg_local_shard_num;
      std::unordered_set<size_t> global_shard_idx;
      std::string global_shard_idx_str;
      for (int j = o_start_idx; j < o_end_idx; ++j) {
        if ((j % _avg_local_shard_num) % _m_real_local_shard_num ==
            m_local_shard_id) {
          global_shard_idx.insert(j);
          global_shard_idx_str.append(std::to_string(j)).append(",");
        }
      }
      try {
        while (read_channel->read_line(line_data) == 0 &&
               line_data.size() > 1) {
          uint64_t key = std::strtoul(line_data.data(), &end, 10);

          auto index_iter =
              global_shard_idx.find(key % _sparse_table_shard_num);
          if (index_iter == global_shard_idx.end()) {
            LOG(WARNING) << "MemorySparseTable key:" << key
                         << " not match shard,"
                         << " file_idx:" << i
                         << " global_shard_idx:" << global_shard_idx_str
                         << " shard num:" << _sparse_table_shard_num
                         << " file:" << channel_config.path;
            continue;
          }
          size_t local_shard_idx = *index_iter % _avg_local_shard_num;
          auto &shard = _local_shards[local_shard_idx];

          auto &value = shard[key];
          value.resize(feature_value_size);
          int parse_size =
              _value_accessor->ParseFromString(++end, value.data());
          value.resize(parse_size);
        }
        read_channel->close();
        if (err_no == -1) {
          ++retry_num;
          is_read_failed = true;
          LOG(ERROR)
              << "MemorySparseTable load failed after read, retry it! path:"
              << channel_config.path << " , retry_num=" << retry_num;
        }
      } catch (...) {
        ++retry_num;
        is_read_failed = true;
        LOG(ERROR) << "MemorySparseTable load failed, retry it! path:"
                   << channel_config.path << " , retry_num=" << retry_num;
      }
      if (retry_num > FLAGS_pserver_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable load failed reach max limit!";
        exit(-1);
      }
    } while (is_read_failed);
  }
  LOG(INFO) << "MemorySparseTable load success, path from "
            << file_list[start_idx] << " to " << file_list[end_idx - 1];
  return 0;
}

void MemorySparseTable::Revert() {
  for (int i = 0; i < _real_local_shard_num; ++i) {
    _local_shards_new[i].clear();
  }
}

void MemorySparseTable::CheckSavePrePatchDone() {
  _save_patch_model_thread.join();
}

int32_t MemorySparseTable::Save(const std::string &dirname,
                                const std::string &param) {
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  // gpu graph mode
  if (_use_gpu_graph) {
    auto *save_filtered_slots = _value_accessor->GetSaveFilteredSlots();
    if (save_filtered_slots != nullptr && (save_filtered_slots->size()) > 0) {
      return Save_v2(dirname, param);
    }
  }
#endif
  if (_real_local_shard_num == 0) {
    _local_show_threshold = -1;
    return 0;
  }

  VLOG(0) << "MemorySparseTable::save dirname: " << dirname;
  int save_param =
      atoi(param.c_str());  // checkpoint:0  xbox delta:1  xbox base:2

  // patch model
  if (save_param == 5) {
    _local_shards_patch_model.reset(_local_shards_new.release());
    _local_shards_new.reset(new shard_type[_real_local_shard_num]);  // NOLINT
    _save_patch_model_thread = std::thread(std::bind(
        &MemorySparseTable::SavePatch, this, std::string(dirname), save_param));
    return 0;
  }

  // cache model
  int64_t tk_size = LocalSize() * _config.sparse_table_cache_rate();
  TopkCalculator tk(_real_local_shard_num, tk_size);

  std::string table_path = TableDir(dirname);
  _afs_client.remove(::paddle::string::format_string(
      "%s/part-%03d-*", table_path.c_str(), _shard_idx));
  std::atomic<uint32_t> feasign_size_all{0};

  size_t file_start_idx = _avg_local_shard_num * _shard_idx;

#ifdef PADDLE_WITH_HETERPS
  int thread_num = _real_local_shard_num;
#else
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;
#endif
  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < _real_local_shard_num; ++i) {
    FsChannelConfig channel_config = {};
    if (_config.compress_in_save() && (save_param == 0 || save_param == 3)) {
      channel_config.path =
          ::paddle::string::format_string("%s/part-%03d-%05d.gz",
                                          table_path.c_str(),
                                          _shard_idx,
                                          file_start_idx + i);
    } else {
      channel_config.path = ::paddle::string::format_string("%s/part-%03d-%05d",
                                                            table_path.c_str(),
                                                            _shard_idx,
                                                            file_start_idx + i);
    }
    channel_config.converter = _value_accessor->Converter(save_param).converter;
    channel_config.deconverter =
        _value_accessor->Converter(save_param).deconverter;
    bool is_write_failed = false;
    int feasign_size = 0;
    int retry_num = 0;
    int err_no = 0;
    auto &shard = _local_shards[i];
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
    // for incremental training, batch_model increase unseenday before save
    if (_use_gpu_graph && save_param == 3) {
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        _value_accessor->UpdateStatAfterSave(it.value().data(), save_param);
      }
    }
#endif
    do {
      err_no = 0;
      feasign_size = 0;
      is_write_failed = false;
      auto write_channel =
          _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        if (_config.enable_sparse_table_cache() &&
            (save_param == 1 || save_param == 2) &&
            _value_accessor->Save(it.value().data(), 4)) {
          CostTimer timer10("sprase table top push");
          tk.push(i, _value_accessor->GetField(it.value().data(), "show"));
        }

        if (_value_accessor->Save(it.value().data(), save_param)) {
          std::string format_value = _value_accessor->ParseToString(
              it.value().data(), it.value().size());
          if (0 != write_channel->write_line(::paddle::string::format_string(
                       "%lu %s", it.key(), format_value.c_str()))) {
            ++retry_num;
            is_write_failed = true;
            LOG(ERROR)
                << "MemorySparseTable save prefix failed, retry it! path:"
                << channel_config.path << " , retry_num=" << retry_num;
            break;
          }
          ++feasign_size;
        }
      }
      write_channel->close();
      if (err_no == -1) {
        ++retry_num;
        is_write_failed = true;
        LOG(ERROR)
            << "MemorySparseTable save prefix failed after write, retry it! "
            << "path:" << channel_config.path << " , retry_num=" << retry_num;
      }
      if (is_write_failed) {
        _afs_client.remove(channel_config.path);
      }
      if (retry_num > FLAGS_pserver_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable save prefix failed reach max limit!";
        exit(-1);
      }
    } while (is_write_failed);
    feasign_size_all += feasign_size;
    if (!_use_gpu_graph) {
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        _value_accessor->UpdateStatAfterSave(it.value().data(), save_param);
      }
    } else if (save_param != 3) {
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        _value_accessor->UpdateStatAfterSave(it.value().data(), save_param);
      }
    }
    LOG(INFO) << "MemorySparseTable save prefix success, path: "
              << channel_config.path << " feasign_size: " << feasign_size;
  }
  _local_show_threshold = tk.top();
  // int32 may overflow need to change return value
  return 0;
}

#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
int32_t MemorySparseTable::Save_v2(const std::string &dirname,
                                   const std::string &param) {
  if (_real_local_shard_num == 0) {
    _local_show_threshold = -1;
    return 0;
  }

  VLOG(0) << "MemorySparseTable::save dirname: " << dirname;
  int save_param =
      atoi(param.c_str());  // checkpoint:0  xbox delta:1  xbox base:2

  // patch model
  if (save_param == 5) {
    _local_shards_patch_model.reset(_local_shards_new.release());
    _local_shards_new.reset(new shard_type[_real_local_shard_num]);
    _save_patch_model_thread = std::thread(std::bind(
        &MemorySparseTable::SavePatch, this, std::string(dirname), save_param));
    return 0;
  }

  // cache model
  int64_t tk_size = LocalSize() * _config.sparse_table_cache_rate();
  TopkCalculator tk(_real_local_shard_num, tk_size);

  std::string table_path = TableDir(dirname);
  _afs_client.remove(::paddle::string::format_string(
      "%s/part-%03d-*", table_path.c_str(), _shard_idx));
  // path to save non 9008 slot's feasign
  _afs_client.remove(paddle::string::format_string(
      "%s/slot_feature/part-%03d-*", table_path.c_str(), _shard_idx));
  std::atomic<uint32_t> feasign_size_all{0};
  std::atomic<uint32_t> feasign_size_all_for_slot_feature{0};

  size_t file_start_idx = _avg_local_shard_num * _shard_idx;

#ifdef PADDLE_WITH_HETERPS
  int thread_num = _real_local_shard_num;
#else
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;
#endif
  omp_set_num_threads(thread_num);

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < _real_local_shard_num; ++i) {
    FsChannelConfig channel_config = {};
    FsChannelConfig channel_config_for_slot_feature;

    if (_config.compress_in_save() && (save_param == 0 || save_param == 3)) {
      channel_config.path =
          paddle::string::format_string("%s/part-%03d-%05d.gz",
                                        table_path.c_str(),
                                        _shard_idx,
                                        file_start_idx + i);
      channel_config_for_slot_feature.path =
          paddle::string::format_string("%s/slot_feature/part-%03d-%05d.gz",
                                        table_path.c_str(),
                                        _shard_idx,
                                        file_start_idx + i);
    } else {
      channel_config.path = paddle::string::format_string("%s/part-%03d-%05d",
                                                          table_path.c_str(),
                                                          _shard_idx,
                                                          file_start_idx + i);
      channel_config_for_slot_feature.path =
          paddle::string::format_string("%s/slot_feature/part-%03d-%05d",
                                        table_path.c_str(),
                                        _shard_idx,
                                        file_start_idx + i);
    }
    channel_config.converter = _value_accessor->Converter(save_param).converter;
    channel_config.deconverter =
        _value_accessor->Converter(save_param).deconverter;
    channel_config_for_slot_feature.converter =
        _value_accessor->Converter(save_param).converter;
    channel_config_for_slot_feature.deconverter =
        _value_accessor->Converter(save_param).deconverter;

    bool is_write_failed = false;
    bool is_write_failed_for_slot_feature = false;
    int feasign_size = 0;
    int feasign_size_for_slot_feature = 0;
    int retry_num = 0;
    int retry_num_for_slot_feature = 0;
    int err_no = 0;
    int err_no_for_slot_feature = 0;
    auto &shard = _local_shards[i];
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
    // for incremental training, batch_model increase unseenday before save
    if (_use_gpu_graph && save_param == 3) {
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        _value_accessor->UpdateStatAfterSave(it.value().data(), save_param);
      }
    }
#endif
    do {
      err_no = 0;
      err_no_for_slot_feature = 0;
      feasign_size = 0;
      feasign_size_for_slot_feature = 0;
      is_write_failed = false;
      auto write_channel =
          _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
      auto write_channel_for_slot_feature =
          _afs_client.open_w(channel_config_for_slot_feature,
                             1024 * 1024 * 40,
                             &err_no_for_slot_feature);

      for (auto it = shard.begin(); it != shard.end(); ++it) {
        if (_config.enable_sparse_table_cache() &&
            (save_param == 1 || save_param == 2) &&
            _value_accessor->Save(it.value().data(), 4)) {
          CostTimer timer10("sparse table top push");
          tk.push(i, _value_accessor->GetField(it.value().data(), "show"));
        }

        if (_value_accessor->Save(it.value().data(), save_param)) {
          std::string format_value = _value_accessor->ParseToString(
              it.value().data(), it.value().size());
          if (0 != write_channel->write_line(::paddle::string::format_string(
                       "%lu %s", it.key(), format_value.c_str()))) {
            ++retry_num;
            is_write_failed = true;
            LOG(ERROR)
                << "MemorySparseTable save prefix failed, retry it! path:"
                << channel_config.path << " , retry_num=" << retry_num;
            break;
          }
          ++feasign_size;
          // save non 9008 slot's feasign
          if (_value_accessor->SaveFilterSlot(it.value().data())) {
            if (0 != write_channel_for_slot_feature->write_line(
                         paddle::string::format_string(
                             "%lu %s", it.key(), format_value.c_str()))) {
              ++retry_num_for_slot_feature;
              is_write_failed_for_slot_feature = true;
              LOG(ERROR) << "MemorySparseTable save slot feature failed, retry "
                            "it! path:"
                         << channel_config_for_slot_feature.path
                         << " , retry_num=" << retry_num_for_slot_feature;
              break;
            }
            ++feasign_size_for_slot_feature;
          }
        }
      }
      write_channel->close();
      write_channel_for_slot_feature->close();
      if (err_no == -1) {
        ++retry_num;
        is_write_failed = true;
        LOG(ERROR)
            << "MemorySparseTable save prefix failed after write, retry it! "
            << "path:" << channel_config.path << " , retry_num=" << retry_num;
      }
      if (err_no_for_slot_feature == -1) {
        ++retry_num_for_slot_feature;
        is_write_failed_for_slot_feature = true;
        LOG(ERROR)
            << "MemorySparseTable save prefix failed after write, retry it! "
            << "path:" << channel_config_for_slot_feature.path
            << " , retry_num=" << retry_num_for_slot_feature;
      }
      if (is_write_failed) {
        _afs_client.remove(channel_config.path);
      }
      if (is_write_failed_for_slot_feature) {
        _afs_client.remove(channel_config_for_slot_feature.path);
      }
      if (retry_num > FLAGS_pserver_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable save prefix failed reach max limit!";
        exit(-1);
      }
      if (retry_num_for_slot_feature > FLAGS_pserver_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable save prefix for slot feature failed "
                      "reach max limit!";
        exit(-1);
      }
    } while (is_write_failed && is_write_failed_for_slot_feature);

    feasign_size_all += feasign_size;
    feasign_size_all_for_slot_feature += feasign_size_for_slot_feature;
    if (!_use_gpu_graph) {
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        _value_accessor->UpdateStatAfterSave(it.value().data(), save_param);
      }
    } else if (save_param != 3) {
      for (auto it = shard.begin(); it != shard.end(); ++it) {
        _value_accessor->UpdateStatAfterSave(it.value().data(), save_param);
      }
    }
    LOG(INFO) << "MemorySparseTable save prefix&feature success, path: "
              << channel_config.path << " feasign_size: " << feasign_size
              << ", feature path:" << channel_config_for_slot_feature.path
              << ", feature feasign size:" << feasign_size_for_slot_feature;
  }
  _local_show_threshold = tk.top();
  // int32 may overflow need to change return value
  return 0;
}
#endif

int32_t MemorySparseTable::SavePatch(const std::string &path, int save_param) {
  if (!_config.enable_revert()) {
    LOG(INFO) << "MemorySparseTable should be enabled revert.";
    return 0;
  }
  size_t file_start_idx = _m_avg_local_shard_num * _shard_idx;
  std::string table_path = TableDir(path);
  _afs_client.remove(::paddle::string::format_string(
      "%s/part-%03d-*", table_path.c_str(), _shard_idx));
  int thread_num = _m_real_local_shard_num < 20 ? _m_real_local_shard_num : 20;

  std::atomic<uint32_t> feasign_size_all{0};

  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < _m_real_local_shard_num; ++i) {
    FsChannelConfig channel_config = {};
    channel_config.path = ::paddle::string::format_string("%s/part-%03d-%05d",
                                                          table_path.c_str(),
                                                          _shard_idx,
                                                          file_start_idx + i);

    channel_config.converter = _value_accessor->Converter(save_param).converter;
    channel_config.deconverter =
        _value_accessor->Converter(save_param).deconverter;

    bool is_write_failed = false;
    int feasign_size = 0;
    int retry_num = 0;
    int err_no = 0;
    do {
      err_no = 0;
      feasign_size = 0;
      is_write_failed = false;
      auto write_channel =
          _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);

      for (int j = 0; j < _real_local_shard_num; ++j) {
        if (j % _m_real_local_shard_num == i) {
          auto &shard = _local_shards_patch_model[j];
          for (auto it = shard.begin(); it != shard.end(); ++it) {
            if (_value_accessor->Save(it.value().data(), save_param)) {
              std::string format_value = _value_accessor->ParseToString(
                  it.value().data(), it.value().size());
              if (0 !=
                  write_channel->write_line(::paddle::string::format_string(
                      "%lu %s", it.key(), format_value.c_str()))) {
                ++retry_num;
                is_write_failed = true;
                LOG(ERROR) << "MemorySparseTable save failed, retry it! path:"
                           << channel_config.path
                           << " , retry_num=" << retry_num;
                break;
              }
              ++feasign_size;
            }
          }
        }
        if (is_write_failed) break;
      }
      write_channel->close();
      if (err_no == -1) {
        ++retry_num;
        is_write_failed = true;
        LOG(ERROR)
            << "MemorySparseTable save patch failed after write, retry it! "
            << "path:" << channel_config.path << " , retry_num=" << retry_num;
      }
      if (is_write_failed) {
        _afs_client.remove(channel_config.path);
      }
      if (retry_num > FLAGS_pserver_table_save_max_retry) {
        LOG(ERROR) << "MemorySparseTable save patch failed reach max limit!";
        exit(-1);
      }
    } while (is_write_failed);
    feasign_size_all += feasign_size;
  }
  LOG(INFO) << "MemorySparseTable save patch success, path:"
            << ::paddle::string::format_string("%s/%03d/part-%03d-",
                                               path.c_str(),
                                               _config.table_id(),
                                               _shard_idx)
            << " from " << file_start_idx << " to "
            << file_start_idx + _m_real_local_shard_num - 1
            << ", feasign size: " << feasign_size_all;
  return 0;
}

int64_t MemorySparseTable::CacheShuffle(
    const std::string &path,
    const std::string &param,
    double cache_threshold,
    std::function<std::future<int32_t>(
        int msg_type, int to_pserver_id, std::string &msg)> send_msg_func,
    ::paddle::framework::Channel<std::pair<uint64_t, std::string>>
        &shuffled_channel,
    const std::vector<Table *> &table_ptrs) {
  LOG(INFO) << "cache shuffle with cache threshold: " << cache_threshold;
  int save_param = atoi(param.c_str());  // batch_model:0  xbox:1
  if (!_config.enable_sparse_table_cache() || cache_threshold < 0) {
    LOG(WARNING)
        << "cache shuffle failed not enable table cache or cache threshold < 0 "
        << _config.enable_sparse_table_cache() << " or " << cache_threshold;
    // return -1;
  }
  int shuffle_node_num = _config.sparse_table_cache_file_num();
  LOG(INFO) << "Table>> shuffle node num is: " << shuffle_node_num;
  // TODO(zhaocaibei123): check shuffle_node_num <= server_node_num
  int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;

  std::vector<
      ::paddle::framework::ChannelWriter<std::pair<uint64_t, std::string>>>
      writers(_real_local_shard_num);
  std::vector<std::vector<std::pair<uint64_t, std::string>>> datas(
      _real_local_shard_num);

  int feasign_size = 0;
  std::vector<::paddle::framework::Channel<std::pair<uint64_t, std::string>>>
      tmp_channels;
  for (int i = 0; i < _real_local_shard_num; ++i) {
    tmp_channels.push_back(
        ::paddle::framework::MakeChannel<std::pair<uint64_t, std::string>>());
  }

  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < _real_local_shard_num; ++i) {
    ::paddle::framework::ChannelWriter<std::pair<uint64_t, std::string>>
        &writer = writers[i];
    writer.Reset(tmp_channels[i].get());

    for (auto table_ptr : table_ptrs) {
      auto value_accessor = table_ptr->GetValueAccessor();
      shard_type *shard_ptr = static_cast<shard_type *>(table_ptr->GetShard(i));

      for (auto it = shard_ptr->begin(); it != shard_ptr->end(); ++it) {
        if (value_accessor->SaveCache(
                it.value().data(), save_param, cache_threshold)) {
          std::string format_value = value_accessor->ParseToString(
              it.value().data(), it.value().size());
          std::pair<uint64_t, std::string> pkv(it.key(), format_value.c_str());
          writer << pkv;
          ++feasign_size;
        }
      }
    }
    writer.Flush();
    writer.channel()->Close();
  }
  // LOG(INFO) << "MemorySparseTable cache KV save success to Channel feasign
  // size: " << feasign_size << " and start sparse cache data shuffle real local
  // shard num: " << _real_local_shard_num;
  std::vector<std::pair<uint64_t, std::string>> local_datas;
  for (int idx_shard = 0; idx_shard < _real_local_shard_num; ++idx_shard) {
    ::paddle::framework::ChannelWriter<std::pair<uint64_t, std::string>>
        &writer = writers[idx_shard];
    auto channel = writer.channel();
    std::vector<std::pair<uint64_t, std::string>> &data = datas[idx_shard];
    std::vector<::paddle::framework::BinaryArchive> ars(shuffle_node_num);
    while (channel->Read(data)) {
      for (auto &t : data) {
        auto pserver_id =
            ::paddle::distributed::local_random_engine()() % shuffle_node_num;
        if (pserver_id != _shard_idx) {
          ars[pserver_id] << t;
        } else {
          local_datas.emplace_back(std::move(t));
        }
      }
      std::vector<std::future<int32_t>> total_status;
      std::vector<uint32_t> send_data_size(shuffle_node_num, 0);
      std::vector<int> send_index(shuffle_node_num);
      for (int i = 0; i < shuffle_node_num; ++i) {
        send_index[i] = i;
      }
      std::random_shuffle(send_index.begin(), send_index.end());
      for (int index = 0; index < shuffle_node_num; ++index) {
        int i = send_index[index];
        if (i == static_cast<int>(_shard_idx)) {
          continue;
        }
        if (ars[i].Length() == 0) {
          continue;
        }
        std::string msg(ars[i].Buffer(), ars[i].Length());
        auto ret = send_msg_func(101, i, msg);
        total_status.push_back(std::move(ret));
        send_data_size[i] += ars[i].Length();
      }
      for (auto &t : total_status) {
        t.wait();
      }
      ars.clear();
      ars = std::vector<::paddle::framework::BinaryArchive>(shuffle_node_num);
      data = std::vector<std::pair<uint64_t, std::string>>();
    }
  }
  shuffled_channel->Write(std::move(local_datas));
  return 0;
}

int32_t MemorySparseTable::SaveCache(
    const std::string &path,
    const std::string &param,
    ::paddle::framework::Channel<std::pair<uint64_t, std::string>>
        &shuffled_channel) {
  if (_shard_idx >= _config.sparse_table_cache_file_num()) {
    return 0;
  }
  int save_param = atoi(param.c_str());  // batch_model:0  xbox:1
  std::string table_path = ::paddle::string::format_string(
      "%s/%03d_cache/", path.c_str(), _config.table_id());
  _afs_client.remove(::paddle::string::format_string(
      "%s/part-%03d", table_path.c_str(), _shard_idx));
  uint32_t feasign_size = 0;
  FsChannelConfig channel_config = {};
  // not compress cache model
  channel_config.path = ::paddle::string::format_string(
      "%s/part-%03d", table_path.c_str(), _shard_idx);
  channel_config.converter = _value_accessor->Converter(save_param).converter;
  channel_config.deconverter =
      _value_accessor->Converter(save_param).deconverter;
  auto write_channel = _afs_client.open_w(channel_config, 1024 * 1024 * 40);
  std::vector<std::pair<uint64_t, std::string>> data;
  bool is_write_failed = false;
  shuffled_channel->Close();
  while (shuffled_channel->Read(data)) {
    for (auto &t : data) {
      ++feasign_size;
      if (0 != write_channel->write_line(::paddle::string::format_string(
                   "%lu %s", t.first, t.second.c_str()))) {
        LOG(ERROR) << "Cache Table save failed, "
                      "path:"
                   << channel_config.path << ", retry it!";
        is_write_failed = true;
        break;
      }
    }
    data = std::vector<std::pair<uint64_t, std::string>>();
  }
  if (is_write_failed) {
    _afs_client.remove(channel_config.path);
  }
  write_channel->close();
  LOG(INFO) << "MemorySparseTable cache save success, feasign: " << feasign_size
            << ", path: " << channel_config.path;
  shuffled_channel->Open();
  return feasign_size;
}

int64_t MemorySparseTable::LocalSize() {
  int64_t local_size = 0;
  for (int i = 0; i < _real_local_shard_num; ++i) {
    local_size += _local_shards[i].size();
  }
  return local_size;
}

int64_t MemorySparseTable::LocalMFSize() {
  std::vector<int64_t> size_arr(_real_local_shard_num, 0);
  std::vector<std::future<int>> tasks(_real_local_shard_num);
  int64_t ret_size = 0;
  for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    tasks[shard_id] =
        _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
            [this, shard_id, &size_arr]() -> int {
              auto &local_shard = _local_shards[shard_id];
              for (auto it = local_shard.begin(); it != local_shard.end();
                   ++it) {
                if (_value_accessor->HasMF(it.value().size())) {
                  size_arr[shard_id] += 1;
                }
              }
              return 0;
            });
  }
  for (int i = 0; i < _real_local_shard_num; ++i) {
    tasks[i].wait();
  }
  for (auto x : size_arr) {
    ret_size += x;
  }
  return ret_size;
}

std::pair<int64_t, int64_t> MemorySparseTable::PrintTableStat() {
  int64_t feasign_size = LocalSize();
  int64_t mf_size = LocalMFSize();
  return {feasign_size, mf_size};
}

int32_t MemorySparseTable::Pull(TableContext &context) {
  PADDLE_ENFORCE_EQ(
      context.value_type,
      Sparse,
      common::errors::InvalidArgument(
          "The 'value_type' in context must be 'Sparse', but received %d.",
          context.value_type));
  if (context.use_ptr) {
    char **pull_values = context.pull_context.ptr_values;
    const uint64_t *keys = context.pull_context.keys;
    return PullSparsePtr(
        context.shard_id, pull_values, keys, context.num, context.pass_id);
  } else {
    float *pull_values = context.pull_context.values;
    const PullSparseValue &pull_value = context.pull_context.pull_value;
    return PullSparse(pull_values, pull_value);
  }
}

int32_t MemorySparseTable::Push(TableContext &context) {
  PADDLE_ENFORCE_EQ(
      context.value_type,
      Sparse,
      common::errors::InvalidArgument(
          "The 'value_type' in context must be 'Sparse', but received %d.",
          context.value_type));
  if (!context.use_ptr) {
    return PushSparse(
        context.push_context.keys, context.push_context.values, context.num);
  } else {
    return PushSparse(context.push_context.keys,
                      context.push_context.ptr_values,
                      context.num);
  }
}

int32_t MemorySparseTable::PullSparse(float *pull_values,
                                      const PullSparseValue &pull_value) {
  CostTimer timer("pserver_sparse_select_all");
  std::vector<std::future<int>> tasks(_real_local_shard_num);

  const size_t value_size =
      _value_accessor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_size =
      _value_accessor->GetAccessorInfo().mf_size / sizeof(float);
  size_t select_value_size =
      _value_accessor->GetAccessorInfo().select_size / sizeof(float);
  // std::atomic<uint32_t> missed_keys{0};

  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      _real_local_shard_num);
  size_t num = pull_value.numel_;
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (pull_value.feasigns_[i] % _sparse_table_shard_num) %
                   _avg_local_shard_num;
    task_keys[shard_id].push_back({pull_value.feasigns_[i], i});
  }
  for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    tasks[shard_id] =
        _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
            [this,
             shard_id,
             &task_keys,
             value_size,
             pull_values,
             mf_value_size,
             select_value_size]() -> int {
              auto &local_shard = _local_shards[shard_id];
              float data_buffer[value_size];  // NOLINT
              float *data_buffer_ptr = data_buffer;

              auto &keys = task_keys[shard_id];
              for (auto &item : keys) {
                uint64_t key = item.first;
                auto itr = local_shard.find(key);
                size_t data_size = value_size - mf_value_size;
                if (itr == local_shard.end()) {
                  // ++missed_keys;
                  if (FLAGS_pserver_create_value_when_push) {
                    memset(data_buffer, 0, sizeof(float) * data_size);
                  } else {
                    auto &feature_value = local_shard[key];
                    feature_value.resize(data_size);
                    float *data_ptr = feature_value.data();
                    _value_accessor->Create(&data_buffer_ptr, 1);
                    memcpy(
                        data_ptr, data_buffer_ptr, data_size * sizeof(float));
                  }
                } else {
                  data_size = itr.value().size();
                  memcpy(data_buffer_ptr,
                         itr.value().data(),
                         data_size * sizeof(float));
                }
                for (size_t mf_idx = data_size; mf_idx < value_size; ++mf_idx) {
                  data_buffer[mf_idx] = 0.0;
                }
                auto offset = item.second;
                float *select_data = pull_values + select_value_size * offset;
                _value_accessor->Select(
                    &select_data, (const float **)&data_buffer_ptr, 1);
              }

              return 0;
            });
  }

  for (auto &task : tasks) {
    task.wait();
  }
  return 0;
}

int32_t MemorySparseTable::PullSparsePtr(int shard_id,  // fake num
                                         char **pull_values,
                                         const uint64_t *keys,
                                         size_t num,
                                         uint16_t pass_id) {
  CostTimer timer("pscore_sparse_select_all");
  size_t value_size = _value_accessor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_size =
      _value_accessor->GetAccessorInfo().mf_size / sizeof(float);

  std::vector<std::future<int>> tasks(_real_local_shard_num);
  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      _real_local_shard_num);
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (keys[i] % _sparse_table_shard_num) % _avg_local_shard_num;
    task_keys[shard_id].push_back({keys[i], i});
  }
  // std::atomic<uint32_t> missed_keys{0};
  for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    tasks[shard_id] =
        _shards_task_pool[shard_id % _shards_task_pool.size()]->enqueue(
            [this,
             shard_id,
             &task_keys,
             pull_values,
             value_size,
             mf_value_size]() -> int {
              auto &keys = task_keys[shard_id];
              auto &local_shard = _local_shards[shard_id];
              float data_buffer[value_size];  // NOLINT
              float *data_buffer_ptr = data_buffer;
              for (auto &item : keys) {
                uint64_t key = item.first;
                auto itr = local_shard.find(key);
                size_t data_size = value_size - mf_value_size;
                FixedFeatureValue *ret = NULL;
                if (itr == local_shard.end()) {
                  // ++missed_keys;
                  auto &feature_value = local_shard[key];
                  feature_value.resize(data_size);
                  float *data_ptr = feature_value.data();
                  _value_accessor->Create(&data_buffer_ptr, 1);
                  memcpy(data_ptr, data_buffer_ptr, data_size * sizeof(float));
                  ret = &feature_value;
                } else {
                  ret = itr.value_ptr();
                }
                int pull_data_idx = item.second;
                pull_values[pull_data_idx] = reinterpret_cast<char *>(ret);
              }
              return 0;
            });
  }
  for (auto &task : tasks) {
    task.wait();
  }
  return 0;
}

int32_t MemorySparseTable::PushSparse(const uint64_t *keys,
                                      const float *values,
                                      size_t num) {
  CostTimer timer("pserver_sparse_update_all");
  std::vector<std::future<int>> tasks(_real_local_shard_num);
  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      _real_local_shard_num);
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (keys[i] % _sparse_table_shard_num) % _avg_local_shard_num;
    task_keys[shard_id].push_back({keys[i], i});
  }

  const size_t value_col =
      _value_accessor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_col =
      _value_accessor->GetAccessorInfo().mf_size / sizeof(float);
  size_t update_value_col =
      _value_accessor->GetAccessorInfo().update_size / sizeof(float);

  for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id % _task_pool_size]->enqueue(
        [this,
         shard_id,
         value_col,
         mf_value_col,
         update_value_col,
         values,
         &task_keys]() -> int {
          auto &keys = task_keys[shard_id];
          auto &local_shard = _local_shards[shard_id];
          auto &local_shard_new = _local_shards_new[shard_id];
          float data_buffer[value_col];  // NOLINT
          float *data_buffer_ptr = data_buffer;
          for (auto &item : keys) {
            uint64_t key = item.first;
            uint64_t push_data_idx = item.second;
            const float *update_data =
                values + push_data_idx * update_value_col;
            auto itr = local_shard.find(key);
            if (itr == local_shard.end()) {
              if (FLAGS_pserver_enable_create_feasign_randomly &&
                  !_value_accessor->CreateValue(1, update_data)) {
                continue;
              }
              auto value_size = value_col - mf_value_col;
              auto &feature_value = local_shard[key];
              feature_value.resize(value_size);
              _value_accessor->Create(&data_buffer_ptr, 1);
              memcpy(feature_value.data(),
                     data_buffer_ptr,
                     value_size * sizeof(float));
              itr = local_shard.find(key);
            }

            auto &feature_value = itr.value();
            float *value_data = feature_value.data();
            size_t value_size = feature_value.size();

            if (value_size == value_col) {  // 已拓展到最大size, 则就地update
              _value_accessor->Update(&value_data, &update_data, 1);
            } else {
              // 拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
              memcpy(data_buffer_ptr, value_data, value_size * sizeof(float));
              _value_accessor->Update(&data_buffer_ptr, &update_data, 1);

              if (_value_accessor->NeedExtendMF(data_buffer)) {
                feature_value.resize(value_col);
                value_data = feature_value.data();
                _value_accessor->Create(&value_data, 1);
              }
              memcpy(value_data, data_buffer_ptr, value_size * sizeof(float));
            }
            if (_config.enable_revert()) {
              FixedFeatureValue *feature_value_new = &(local_shard_new[key]);
              auto new_size = feature_value.size();
              feature_value_new->resize(new_size);
              memcpy(feature_value_new->data(),
                     value_data,
                     new_size * sizeof(float));
            }
          }
          return 0;
        });
  }

  for (auto &task : tasks) {
    task.wait();
  }
  return 0;
}

int32_t MemorySparseTable::PushSparse(const uint64_t *keys,
                                      const float **values,
                                      size_t num) {
  std::vector<std::future<int>> tasks(_real_local_shard_num);
  std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(
      _real_local_shard_num);
  for (size_t i = 0; i < num; ++i) {
    int shard_id = (keys[i] % _sparse_table_shard_num) % _avg_local_shard_num;
    task_keys[shard_id].push_back({keys[i], i});
  }

  size_t value_col = _value_accessor->GetAccessorInfo().size / sizeof(float);
  size_t mf_value_col =
      _value_accessor->GetAccessorInfo().mf_size / sizeof(float);

  for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id % _task_pool_size]->enqueue(
        [this, shard_id, value_col, mf_value_col, values, &task_keys]() -> int {
          auto &keys = task_keys[shard_id];
          auto &local_shard = _local_shards[shard_id];
          float data_buffer[value_col];  // NOLINT
          float *data_buffer_ptr = data_buffer;
          for (auto &item : keys) {
            uint64_t key = item.first;
            uint64_t push_data_idx = item.second;
            const float *update_data = values[push_data_idx];
            auto itr = local_shard.find(key);
            if (itr == local_shard.end()) {
              if (FLAGS_pserver_enable_create_feasign_randomly &&
                  !_value_accessor->CreateValue(1, update_data)) {
                continue;
              }
              auto value_size = value_col - mf_value_col;
              auto &feature_value = local_shard[key];
              feature_value.resize(value_size);
              _value_accessor->Create(&data_buffer_ptr, 1);
              memcpy(feature_value.data(),
                     data_buffer_ptr,
                     value_size * sizeof(float));
              itr = local_shard.find(key);
            }
            auto &feature_value = itr.value();
            float *value_data = feature_value.data();
            size_t value_size = feature_value.size();
            if (value_size == value_col) {  // 已拓展到最大size, 则就地update
              _value_accessor->Update(&value_data, &update_data, 1);
            } else {
              // 拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
              memcpy(data_buffer_ptr, value_data, value_size * sizeof(float));
              _value_accessor->Update(&data_buffer_ptr, &update_data, 1);
              if (_value_accessor->NeedExtendMF(data_buffer)) {
                feature_value.resize(value_col);
                value_data = feature_value.data();
                _value_accessor->Create(&value_data, 1);
              }
              memcpy(value_data, data_buffer_ptr, value_size * sizeof(float));
            }
          }
          return 0;
        });
  }

  for (auto &task : tasks) {
    task.wait();
  }
  return 0;
}

int32_t MemorySparseTable::Flush() { return 0; }

int32_t MemorySparseTable::Shrink(const std::string &param) {
  VLOG(0) << "MemorySparseTable::Shrink";
  std::atomic<uint32_t> shrink_size_all{0};
  int thread_num = _real_local_shard_num;
  omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic)
  for (int shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
    // Shrink
    int feasign_size = 0;
    auto &shard = _local_shards[shard_id];
    for (auto it = shard.begin(); it != shard.end();) {
      if (_value_accessor->Shrink(it.value().data())) {
        it = shard.erase(it);
        ++feasign_size;
      } else {
        ++it;
      }
    }
    shrink_size_all += feasign_size;
  }
  VLOG(0) << "MemorySparseTable::Shrink success, shrink size:"
          << shrink_size_all;
  return 0;
}

void MemorySparseTable::Clear() { VLOG(0) << "clear coming soon"; }

}  // namespace paddle::distributed
