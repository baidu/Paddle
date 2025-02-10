/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#pragma once

#include <ThreadPool.h>

#include <fstream>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/common/macros.h"
#ifdef PADDLE_WITH_GLOO
#include <gloo/broadcast.h>

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

// Dataset is a abstract class, which defines user interfaces
// Example Usage:
//    Dataset* dataset = DatasetFactory::CreateDataset("InMemoryDataset")
//    dataset->SetFileList(std::vector<std::string>{"a.txt", "b.txt"})
//    dataset->SetThreadNum(1)
//    dataset->CreateReaders();
//    dataset->SetDataFeedDesc(your_data_feed_desc);
//    dataset->LoadIntoMemory();
//    dataset->SetTrainerNum(2);
//    dataset->GlobalShuffle();
class Dataset {
 public:
  Dataset() {}
  virtual ~Dataset() {}
  // do sample
  virtual void TDMSample(const std::string tree_name UNUSED,
                         const std::string tree_path UNUSED,
                         const std::vector<uint16_t> tdm_layer_counts UNUSED,
                         const uint16_t start_sample_layer UNUSED,
                         const bool with_hierarchy UNUSED,
                         const uint16_t seed_ UNUSED,
                         const uint16_t sample_slot UNUSED) {}
  // set file list
  virtual void SetFileList(const std::vector<std::string>& filelist) = 0;
  // set readers' num
  virtual void SetThreadNum(int thread_num) = 0;
  // set workers' num
  virtual void SetTrainerNum(int trainer_num) = 0;
  // set fleet send batch size
  virtual void SetFleetSendBatchSize(int64_t size) = 0;
  virtual void ReleaseMemoryFun() = 0;
  // set fs name and ugi
  virtual void SetHdfsConfig(const std::string& fs_name,
                             const std::string& fs_ugi) = 0;
  // set customized download command, such as using afs api
  virtual void SetDownloadCmd(const std::string& download_cmd) = 0;
  // set data fedd desc, which contains:
  //   data feed name, batch size, slots
  virtual void SetDataFeedDesc(const std::string& data_feed_desc_str) = 0;
  // set channel num
  virtual void SetChannelNum(int channel_num) = 0;
  // set parse ins id
  virtual void SetParseInsId(bool parse_ins_id) = 0;
  virtual void SetParseContent(bool parse_content) = 0;
  virtual void SetParseLogKey(bool parse_logkey) = 0;
  virtual void SetEnablePvMerge(bool enable_pv_merge) = 0;
  virtual bool EnablePvMerge() = 0;
  virtual void SetMergeBySid(bool is_merge) = 0;
  virtual void SetShuffleByUid(bool enable_shuffle_uid) = 0;
  // set merge by ins id
  virtual void SetMergeByInsId(int merge_size) = 0;
  virtual void SetGenerateUniqueFeasign(bool gen_uni_feasigns) = 0;
  // set fea eval mode
  virtual void SetFeaEval(bool fea_eval, int record_candidate_size) = 0;
  // get file list
  virtual const std::vector<std::string>& GetFileList() = 0;
  // get thread num
  virtual int GetThreadNum() = 0;
  // get worker num
  virtual int GetTrainerNum() = 0;
  // get fleet send batch size
  virtual int64_t GetFleetSendBatchSize() = 0;
  // get hdfs config
  virtual std::pair<std::string, std::string> GetHdfsConfig() = 0;
  // get download cmd
  virtual std::string GetDownloadCmd() = 0;
  // get data fedd desc
  virtual const paddle::framework::DataFeedDesc& GetDataFeedDesc() = 0;
  // get channel num
  virtual int GetChannelNum() = 0;
  // get readers, the reader num depend both on thread num
  // and filelist size
  virtual std::vector<paddle::framework::DataFeed*> GetReaders() = 0;
  // create input channel and output channel
  virtual void CreateChannel() = 0;
  // register message handler between workers
  virtual void RegisterClientToClientMsgHandler() = 0;
  // load all data into memory
  virtual void LoadIntoMemory() = 0;
  // load all data into memory in async mode
  virtual void PreLoadIntoMemory() = 0;
  // wait async load done
  virtual void WaitPreLoadDone() = 0;
  // release all memory data
  virtual void ReleaseMemory() = 0;
  // local shuffle data
  virtual void LocalShuffle() = 0;
  // global shuffle data
  virtual void GlobalShuffle(int thread_num = -1) = 0;
  virtual void SlotsShuffle(const std::set<std::string>& slots_to_replace) = 0;
  // create readers
  virtual void CreateReaders() = 0;
  // destroy readers
  virtual void DestroyReaders() = 0;
  // get memory data size
  virtual int64_t GetMemoryDataSize() = 0;
  // get memory data size in input_pv_channel_
  virtual int64_t GetPvDataSize() = 0;
  // get shuffle data size
  virtual int64_t GetShuffleDataSize() = 0;
  // merge by ins id
  virtual void MergeByInsId() = 0;
  // merge pv instance
  virtual void PreprocessInstance() = 0;
  // divide pv instance
  virtual void PostprocessInstance() = 0;
  // only for untest
  virtual void SetCurrentPhase(int current_phase) = 0;
  virtual void GenerateLocalTablesUnlock(int table_id,
                                         int feadim,
                                         int read_thread_num,
                                         int consume_thread_num,
                                         int shard_num) = 0;
  virtual void ClearLocalTables() = 0;
  // create preload readers
  virtual void CreatePreLoadReaders() = 0;
  // destroy preload readers after preload done
  virtual void DestroyPreLoadReaders() = 0;
  // set preload thread num
  virtual void SetPreLoadThreadNum(int thread_num) = 0;
  // separate train thread and dataset thread
  virtual void DynamicAdjustChannelNum(int channel_num,
                                       bool discard_remaining_ins = false) = 0;
  virtual void DynamicAdjustReadersNum(int thread_num) = 0;
  // set fleet send sleep seconds
  virtual void SetFleetSendSleepSeconds(int seconds) = 0;

  virtual std::vector<std::string> GetSlots() = 0;

  virtual void SetGpuGraphMode(int is_graph_mode) = 0;
  virtual int GetGpuGraphMode() = 0;
  virtual bool GetEpochFinish() = 0;
  virtual void ClearSampleState() = 0;

  virtual void SetPassId(uint32_t pass_id) = 0;
  virtual uint32_t GetPassID() = 0;

  virtual void DumpWalkPath(std::string dump_path, size_t dump_rate) = 0;
  virtual void DumpSampleNeighbors(std::string dump_path) = 0;
  virtual const std::vector<uint64_t>& GetGpuGraphTotalKeys() = 0;
  virtual const std::vector<std::vector<uint64_t>*>& GetPassKeysVec() = 0;
  virtual const std::vector<std::vector<uint32_t>*>& GetPassRanksVec() = 0;
  virtual const std::vector<std::shared_ptr<HashTable<uint64_t, uint32_t>>>
  GetPassKeys2RankTable() = 0;

 protected:
  virtual int ReceiveFromClient(int msg_type,
                                int client_id,
                                const std::string& msg) = 0;
};

// DatasetImpl is the implementation of Dataset,
// it holds memory data if user calls load_into_memory
template <typename T>
class DatasetImpl : public Dataset {
 public:
  DatasetImpl();
  virtual ~DatasetImpl() {
    if (release_thread_ != nullptr) {
      release_thread_->join();
    }
  }
  virtual void SetFileList(const std::vector<std::string>& filelist);
  virtual void ReleaseMemoryFun();
  virtual void SetThreadNum(int thread_num);
  virtual void SetTrainerNum(int trainer_num);
  virtual void SetFleetSendBatchSize(int64_t size);
  virtual void SetHdfsConfig(const std::string& fs_name,
                             const std::string& fs_ugi);
  virtual void SetDownloadCmd(const std::string& download_cmd);
  virtual void SetDataFeedDesc(const std::string& data_feed_desc_str);
  virtual void SetChannelNum(int channel_num);
  virtual void SetParseInsId(bool parse_ins_id);
  virtual void SetParseContent(bool parse_content);
  virtual void SetParseLogKey(bool parse_logkey);
  virtual void SetEnablePvMerge(bool enable_pv_merge);
  virtual void SetMergeBySid(bool is_merge);
  virtual void SetShuffleByUid(bool enable_shuffle_uid);

  virtual void SetMergeByInsId(int merge_size);
  virtual void SetGenerateUniqueFeasign(bool gen_uni_feasigns);
  virtual void SetFeaEval(bool fea_eval, int record_candidate_size);
  virtual const std::vector<std::string>& GetFileList() { return filelist_; }
  virtual int GetThreadNum() { return thread_num_; }
  virtual int GetTrainerNum() { return trainer_num_; }
  virtual Channel<T> GetInputChannel() { return input_channel_; }
  virtual void SetInputChannel(const Channel<T>& input_channel) {
    input_channel_ = input_channel;
  }
  virtual int64_t GetFleetSendBatchSize() { return fleet_send_batch_size_; }
  virtual std::pair<std::string, std::string> GetHdfsConfig() {
    return std::make_pair(fs_name_, fs_ugi_);
  }
  virtual void SetGpuGraphMode(int is_graph_mode);
  virtual int GetGpuGraphMode();
  virtual std::string GetDownloadCmd();
  virtual const paddle::framework::DataFeedDesc& GetDataFeedDesc() {
    return data_feed_desc_;
  }
  virtual int GetChannelNum() { return channel_num_; }
  virtual bool EnablePvMerge() { return enable_pv_merge_; }
  virtual std::vector<paddle::framework::DataFeed*> GetReaders();
  virtual void CreateChannel();
  virtual void RegisterClientToClientMsgHandler();
  virtual void LoadIntoMemory();
  virtual void PreLoadIntoMemory();
  virtual void WaitPreLoadDone();
  virtual void ReleaseMemory();
  virtual void LocalShuffle();
  virtual void GlobalShuffle(int thread_num UNUSED = -1) {}
  virtual void SlotsShuffle(
      const std::set<std::string>& slots_to_replace UNUSED) {}
  virtual const std::vector<T>& GetSlotsOriginalData() {
    return slots_shuffle_original_data_;
  }
  virtual void CreateReaders();
  virtual void DestroyReaders();
  virtual int64_t GetMemoryDataSize();
  virtual int64_t GetPvDataSize();
  virtual int64_t GetShuffleDataSize();
  virtual void MergeByInsId() {}
  virtual void PreprocessInstance() {}
  virtual void PostprocessInstance() {}
  virtual void SetCurrentPhase(int current_phase UNUSED) {}
  virtual void GenerateLocalTablesUnlock(int table_id UNUSED,
                                         int feadim UNUSED,
                                         int read_thread_num UNUSED,
                                         int consume_thread_num UNUSED,
                                         int shard_num UNUSED) {}
  virtual void ClearLocalTables() {}
  virtual void CreatePreLoadReaders();
  virtual void DestroyPreLoadReaders();
  virtual void SetPreLoadThreadNum(int thread_num);
  virtual void DynamicAdjustChannelNum(int channel_num,
                                       bool discard_remaining_ins = false);
  virtual void DynamicAdjustReadersNum(int thread_num);
  virtual void SetFleetSendSleepSeconds(int seconds);
  virtual std::vector<std::string> GetSlots();
  virtual bool GetEpochFinish();
  virtual void ClearSampleState();
  virtual void DumpWalkPath(std::string dump_path, size_t dump_rate);
  virtual void DumpSampleNeighbors(std::string dump_path);

  std::vector<paddle::framework::Channel<T>>& GetMultiOutputChannel() {
    return multi_output_channel_;
  }

  std::vector<paddle::framework::Channel<T>>& GetCurOutputChannel() {
    if (cur_channel_ == 0) {
      return multi_output_channel_;
    } else {
      return multi_consume_channel_;
    }
  }
  virtual const std::vector<uint64_t>& GetGpuGraphTotalKeys() {
    return gpu_graph_total_keys_;
  }
  virtual const std::vector<std::vector<uint64_t>*>& GetPassKeysVec() {
    return keys_vec_;
  }
  virtual const std::vector<std::vector<uint32_t>*>& GetPassRanksVec() {
    return ranks_vec_;
  }
  virtual const std::vector<std::shared_ptr<HashTable<uint64_t, uint32_t>>>
  GetPassKeys2RankTable() {
    return keys2rank_tables_;
  }

  virtual void SetPassId(uint32_t pass_id) { pass_id_ = pass_id; }
  virtual uint32_t GetPassID() { return pass_id_; }

 protected:
  virtual int ReceiveFromClient(int msg_type UNUSED,
                                int client_id UNUSED,
                                const std::string& msg UNUSED) {
    // TODO(yaoxuefeng) for SlotRecordDataset
    return -1;
  }
  std::vector<std::shared_ptr<paddle::framework::DataFeed>> readers_;
  std::vector<std::shared_ptr<paddle::framework::DataFeed>> preload_readers_;
  paddle::framework::Channel<T> input_channel_;
  paddle::framework::Channel<PvInstance> input_pv_channel_;
  std::vector<paddle::framework::Channel<PvInstance>> multi_pv_output_;
  std::vector<paddle::framework::Channel<PvInstance>> multi_pv_consume_;

  int channel_num_;
  std::vector<paddle::framework::Channel<T>> multi_output_channel_;
  std::vector<paddle::framework::Channel<T>> multi_consume_channel_;
  std::vector<std::unordered_set<uint64_t>> local_tables_;
  // when read ins, we put ins from one channel to the other,
  // and when finish reading, we set cur_channel = 1 - cur_channel,
  // so if cur_channel=0, all data are in output_channel, else consume_channel
  int cur_channel_;
  std::vector<T> slots_shuffle_original_data_;
  RecordCandidateList slots_shuffle_rclist_;
  int thread_num_;
  int pull_sparse_to_local_thread_num_;
  paddle::framework::DataFeedDesc data_feed_desc_;
  int trainer_num_;
  std::vector<std::string> filelist_;
  size_t file_idx_;
  uint64_t total_fea_num_;
  std::mutex mutex_for_pick_file_;
  std::mutex mutex_for_fea_num_;
  std::string fs_name_;
  std::string fs_ugi_;
  int64_t fleet_send_batch_size_;
  int64_t fleet_send_sleep_seconds_;
  std::vector<std::thread> preload_threads_;
  std::thread* release_thread_ = nullptr;
  bool merge_by_ins_id_;
  bool parse_ins_id_;
  bool parse_content_;
  bool parse_logkey_;
  bool merge_by_sid_;
  bool shuffle_by_uid_;
  bool parse_uid_;
  bool enable_pv_merge_;  // True means to merge pv
  int current_phase_;     // 1 join, 0 update
  size_t merge_size_;
  bool slots_shuffle_fea_eval_ = false;
  bool gen_uni_feasigns_ = false;
  int preload_thread_num_;
  std::mutex global_index_mutex_;
  int64_t global_index_ = 0;
  std::vector<std::shared_ptr<ThreadPool>> consume_task_pool_;
  std::vector<T> input_records_;  // only for paddleboxdatafeed
  std::vector<std::string> use_slots_;
  bool enable_heterps_ = false;
  int gpu_graph_mode_ = 0;
  std::vector<uint64_t> gpu_graph_total_keys_;
  typedef std::vector<uint64_t> KEYS;
  typedef std::vector<uint32_t> RANKS;
  std::vector<KEYS*> keys_vec_;
  std::vector<RANKS*> ranks_vec_;
  // keys: key id
  // value: dest machine rank id
  // vector: refer to multi gpu card.
  std::vector<std::shared_ptr<HashTable<uint64_t, uint32_t>>> keys2rank_tables_;
  uint32_t pass_id_ = 0;
};

// use std::vector<MultiSlotType> or Record as data type
class MultiSlotDataset : public DatasetImpl<Record> {
 public:
  MultiSlotDataset() {}
  virtual void TDMSample(const std::string tree_name,
                         const std::string tree_path,
                         const std::vector<uint16_t> tdm_layer_counts,
                         const uint16_t start_sample_layer,
                         const bool with_hierarchy,
                         const uint16_t seed_,
                         const uint16_t sample_slot);
  virtual void MergeByInsId();
  virtual void PreprocessInstance();
  virtual void PostprocessInstance();
  virtual void SetCurrentPhase(int current_phase);
  virtual void GenerateLocalTablesUnlock(int table_id,
                                         int feadim,
                                         int read_thread_num,
                                         int consume_thread_num,
                                         int shard_num);
  virtual void ClearLocalTables() {
    for (auto& t : local_tables_) {
      t.clear();
      std::unordered_set<uint64_t>().swap(t);
    }
    std::vector<std::unordered_set<uint64_t>>().swap(local_tables_);
  }
  virtual void PreprocessChannel(
      const std::set<std::string>& slots_to_replace,
      std::unordered_set<uint16_t>& index_slot);  // NOLINT
  virtual void SlotsShuffle(const std::set<std::string>& slots_to_replace);
  virtual void GetRandomData(
      const std::unordered_set<uint16_t>& slots_to_replace,
      std::vector<Record>* result);
  virtual ~MultiSlotDataset() {}
  virtual void GlobalShuffle(int thread_num = -1);
  virtual void DynamicAdjustReadersNum(int thread_num);
  virtual void PrepareTrain();

 protected:
  virtual int ReceiveFromClient(int msg_type,
                                int client_id,
                                const std::string& msg);
};
class SlotRecordDataset : public DatasetImpl<SlotRecord> {
 public:
  SlotRecordDataset() { SlotRecordPool(); }
  virtual ~SlotRecordDataset() {}
  // create input channel
  virtual void CreateChannel();
  // create readers
  virtual void CreateReaders();
  // release memory
  virtual void ReleaseMemory();
  virtual void GlobalShuffle(int thread_num = -1);
  virtual void DynamicAdjustChannelNum(int channel_num,
                                       bool discard_remaining_ins);
  virtual void PrepareTrain();
  virtual void DynamicAdjustReadersNum(int thread_num);
  void DynamicAdjustBatchNum();

 protected:
  bool enable_heterps_ = true;
};

}  // namespace framework
}  // namespace paddle
