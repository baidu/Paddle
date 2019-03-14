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

#include <fstream>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

class Dataset {
 public:
  Dataset() {}
  virtual ~Dataset() {}
  virtual void SetFileList(const std::vector<std::string>& filelist) = 0;
  virtual void SetThreadNum(int thread_num) = 0;
  virtual void SetTrainerNum(int trainer_num) = 0;
  virtual void SetDataFeedDesc(const std::string& data_feed_desc_str) = 0;
  virtual const std::vector<std::string>& GetFileList() = 0;
  virtual int GetThreadNum() = 0;
  virtual int GetTrainerNum() = 0;
  virtual const paddle::framework::DataFeedDesc& GetDataFeedDesc() = 0;
  virtual std::vector<std::shared_ptr<paddle::framework::DataFeed>>&
  GetReaders() = 0;
  virtual void LoadIntoMemory() = 0;
  virtual void LocalShuffle() = 0;
  virtual void GlobalShuffle() = 0;
  virtual void CreateReaders() = 0;
  virtual void DestroyReaders() = 0;

 protected:
  virtual int ReceiveFromClient(int msg_type, int client_id,
                                const std::string& msg) = 0;
};

template <typename T>
class DatasetImpl : public Dataset {
 public:
  DatasetImpl();
  virtual ~DatasetImpl() {}

  virtual void SetFileList(const std::vector<std::string>& filelist);
  virtual void SetThreadNum(int thread_num);
  virtual void SetTrainerNum(int trainer_num);
  virtual void SetDataFeedDesc(const std::string& data_feed_desc_str);

  virtual const std::vector<std::string>& GetFileList() { return filelist_; }
  virtual int GetThreadNum() { return thread_num_; }
  virtual int GetTrainerNum() { return trainer_num_; }
  virtual const paddle::framework::DataFeedDesc& GetDataFeedDesc() {
    return data_feed_desc_;
  }

  virtual std::vector<std::shared_ptr<paddle::framework::DataFeed>>&
  GetReaders();
  virtual void LoadIntoMemory();
  virtual void LocalShuffle();
  virtual void GlobalShuffle();
  virtual void CreateReaders();
  virtual void DestroyReaders();

 protected:
  virtual int ReceiveFromClient(int msg_type, int client_id,
                                const std::string& msg);
  std::vector<std::shared_ptr<paddle::framework::DataFeed>> readers_;
  std::vector<T> memory_data_;
  std::mutex mutex_for_update_memory_data_;
  int thread_num_;
  paddle::framework::DataFeedDesc data_feed_desc_;
  std::vector<std::string> filelist_;
  int trainer_num_;
};

class MultiSlotDataset : public DatasetImpl<std::vector<MultiSlotType>> {
 public:
  MultiSlotDataset() {}
  virtual ~MultiSlotDataset() {}
};

}  // end namespace framework
}  // end namespace paddle
