// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef PADDLE_WITH_HETERPS

#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"

#include <algorithm>
#include <deque>
#include <unordered_set>

#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#endif
#include "paddle/phi/core/platform/timer.h"
#if defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/distributed/ps/table/depends/feature_value.h"
#endif
#include "paddle/common/flags.h"

COMMON_DECLARE_int32(gpugraph_dedup_pull_push_mode);
COMMON_DECLARE_int32(gpugraph_storage_mode);
COMMON_DECLARE_bool(query_dest_rank_by_multi_node);
COMMON_DECLARE_string(graph_edges_split_mode);

namespace paddle::framework {

#ifdef PADDLE_WITH_PSLIB
void AfsWrapper::init(const std::string& fs_name,
                      const std::string& fs_user,
                      const std::string& pass_wd,
                      const std::string& conf) {
  int ret = afs_handler_.init(
      fs_name.c_str(), fs_user.c_str(), pass_wd.c_str(), conf.c_str());
  if (ret != 0) {
    LOG(ERROR) << "AFS Init Error";
  }
}

int AfsWrapper::remove(const std::string& path) {
  return afs_handler_.remove(path);
}

int AfsWrapper::mkdir(const std::string& path) {
  return afs_handler_.mkdir(path);
}

std::vector<std::string> AfsWrapper::list(const std::string& path) {
  return afs_handler_.list(path);
}

int AfsWrapper::exist(const std::string& path) {
  return afs_handler_.exist(path);
}

int AfsWrapper::upload(const std::string& local_file,
                       const std::string& afs_file) {
  return afs_handler_.upload_file(local_file, afs_file);
}

int AfsWrapper::download(const std::string& local_file,
                         const std::string& afs_file) {
  return afs_handler_.download_file(local_file, afs_file);
}

int AfsWrapper::touchz(const std::string& path) {
  return afs_handler_.touchz(path);
}

std::string AfsWrapper::cat(const std::string& path) {
  return afs_handler_.cat(path);
}

int AfsWrapper::mv(const std::string& old_path, const std::string& dest_path) {
  return afs_handler_.mv(old_path, dest_path);
}
#endif

std::shared_ptr<PSGPUWrapper> PSGPUWrapper::s_instance_ = NULL;
bool PSGPUWrapper::is_initialized_ = false;
std::mutex PSGPUWrapper::ins_mutex;
#ifdef PADDLE_WITH_PSLIB
void PSGPUWrapper::InitAfsApi(const std::string& fs_name,
                              const std::string& fs_user,
                              const std::string& pass_wd,
                              const std::string& conf) {
  int ret = afs_handler_.init(
      fs_name.c_str(), fs_user.c_str(), pass_wd.c_str(), conf.c_str());
  if (ret != 0) {
    VLOG(0) << "AFS Init Error";
  }
  use_afs_api_ = 1;
}
#endif

#if defined(PADDLE_WITH_PSCORE)
void PSGPUWrapper::InitAfsApi(const std::string& fs_name,
                              const std::string& fs_user,
                              const std::string& pass_wd,
                              const std::string& conf) {
  int ret = afs_handle_.Init(fs_name, fs_user, pass_wd, conf);
  if (ret != 0) {
    VLOG(0) << "AFS Init Error";
    return;
  }
  use_afs_api_ = 1;
}
#endif

void PSGPUWrapper::add_key_to_local(const std::vector<uint64_t>& vec_data) {
  size_t total_len = vec_data.size();
  size_t len_per_thread = total_len / thread_keys_thread_num_;
  size_t begin = 0;
  std::vector<std::thread> threads;

  int remain = total_len % thread_keys_thread_num_;
  auto gen_graph_data_func = [this](const std::vector<uint64_t>& total_data,
                                    int begin_index,
                                    int end_index,
                                    int i) {
    for (auto iter = total_data.begin() + begin_index;
         iter != total_data.begin() + end_index;
         iter++) {
      uint64_t cur_key = *iter;
      int shard_id = cur_key % thread_keys_shard_num_;
      this->thread_keys_[i][shard_id].insert(cur_key);
    }
  };
  auto gen_graph_dynamic_mf_func = [this](
                                       const std::vector<uint64_t>& total_data,
                                       int begin_index,
                                       int end_index,
                                       int i) {
    for (auto iter = total_data.begin() + begin_index;
         iter != total_data.begin() + end_index;
         iter++) {
      uint64_t cur_key = *iter;
      int shard_id = cur_key % thread_keys_shard_num_;
      // TODO(lxsbupt): feasign <-> slot <-> multi_dim
      this->thread_dim_keys_[i][shard_id][0].insert(cur_key);
    }
  };
  for (int i = 0; i < thread_keys_thread_num_; i++) {
    if (!multi_mf_dim_) {
      threads.push_back(
          std::thread(gen_graph_data_func,
                      std::ref(vec_data),
                      begin,
                      begin + len_per_thread + (i < remain ? 1 : 0),
                      i));
    } else {
      threads.push_back(
          std::thread(gen_graph_dynamic_mf_func,
                      std::ref(vec_data),
                      begin,
                      begin + len_per_thread + (i < remain ? 1 : 0),
                      i));
    }
    begin += len_per_thread + (i < remain ? 1 : 0);
  }
  for (std::thread& t : threads) {
    t.join();
  }
}

void PSGPUWrapper::add_key_to_gputask(std::shared_ptr<HeterContext> gpu_task) {
  std::vector<std::thread> threads;
  platform::Timer timeline;
  timeline.Start();
  // merge thread_keys to shard_keys
  auto merge_ins_dynamic_mf_func = [this, gpu_task](int shard_num, int dim_id) {
    for (int i = 0; i < thread_keys_thread_num_; ++i) {
      gpu_task->batch_add_keys(
          shard_num, dim_id, thread_dim_keys_[i][shard_num][dim_id]);
      thread_dim_keys_[i][shard_num][dim_id].clear();
    }
  };
  for (int i = 0; i < thread_keys_shard_num_; ++i) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      threads.push_back(std::thread(merge_ins_dynamic_mf_func, i, j));
    }
  }
  for (auto& t : threads) {
    t.join();
  }
  timeline.Pause();

  VLOG(1) << "GpuPs task add keys cost " << timeline.ElapsedSec()
          << " seconds.";
  timeline.Start();
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  if (gpu_graph_mode_) {
    size_t slot_num = static_cast<size_t>(slot_num_for_pull_feature_);
    // no slot_fea mode and whole_hbm mode, only keep one unique_sort action
    if (slot_num > 0 && FLAGS_gpugraph_storage_mode !=
                            paddle::framework::GpuGraphStorageMode::WHOLE_HBM) {
      gpu_task->UniqueKeys();
    }
  }
#endif
#ifdef PADDLE_WITH_PSLIB
  gpu_task->UniqueKeys();
#endif
  timeline.Pause();
  VLOG(1) << "GpuPs task unique cost " << timeline.ElapsedSec() << " seconds.";
}

void PSGPUWrapper::resize_gputask(std::shared_ptr<HeterContext> gpu_task) {
  for (int i = 0; i < thread_keys_shard_num_; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      if (i == 0 && j == multi_mf_dim_ - 1) {
        gpu_task->feature_dim_keys_[i][j].push_back(0);
      }
      gpu_task->value_dim_ptr_[i][j].resize(
          gpu_task->feature_dim_keys_[i][j].size());
    }
  }
}

void PSGPUWrapper::PreBuildTask(std::shared_ptr<HeterContext> gpu_task,
                                Dataset* dataset_for_pull) {
  VLOG(3) << "PSGPUWrapper::BuildGPUPSTask begin";
  platform::Timer timeline;
  timeline.Start();
  int device_num = heter_devices_.size();
  gpu_task->init(thread_keys_shard_num_, device_num, multi_mf_dim_);

  std::vector<std::thread> threads;
  // data should be in input channel

  thread_dim_keys_.resize(thread_keys_thread_num_);
  for (int i = 0; i < thread_keys_thread_num_; i++) {
    thread_dim_keys_[i].resize(thread_keys_shard_num_);
    for (int j = 0; j < thread_keys_shard_num_; j++) {
      thread_dim_keys_[i][j].resize(multi_mf_dim_);
    }
  }

  size_t total_len = 0;
  size_t len_per_thread = 0;
  int remain = 0;
  size_t begin = 0;

  std::string data_set_name = std::string(typeid(*dataset_).name());

  VLOG(1) << "gpu_graph_mode_:" << gpu_graph_mode_;
  if (!gpu_graph_mode_) {
    if (data_set_name.find("SlotRecordDataset") != std::string::npos) {
      VLOG(0) << "ps_gpu_wrapper use SlotRecordDataset";
      SlotRecordDataset* dataset =
          reinterpret_cast<SlotRecordDataset*>(dataset_);
      auto input_channel = dataset->GetInputChannel();
      VLOG(0) << "psgpu wrapperinputslotchannle size: "
              << input_channel->Size();
      const std::deque<SlotRecord>& vec_data = input_channel->GetData();
      total_len = vec_data.size();
      len_per_thread = total_len / thread_keys_thread_num_;
      remain = total_len % thread_keys_thread_num_;
      VLOG(0) << "total len: " << total_len;
      auto gen_dynamic_mf_func = [this](
                                     const std::deque<SlotRecord>& total_data,
                                     int begin_index,
                                     int end_index,
                                     int i) {
        for (auto iter = total_data.begin() + begin_index;
             iter != total_data.begin() + end_index;
             iter++) {
          const auto& ins = *iter;
          const auto& feasign_v = ins->slot_uint64_feasigns_.slot_values;
          const auto& slot_offset = ins->slot_uint64_feasigns_.slot_offsets;
          for (size_t slot_idx = 0; slot_idx < slot_offset_vector_.size();
               slot_idx++) {
            for (size_t j = slot_offset[slot_offset_vector_[slot_idx]];
                 j < slot_offset[slot_offset_vector_[slot_idx] + 1];
                 j++) {
              int shard_id = feasign_v[j] % thread_keys_shard_num_;
              int dim_id = slot_index_vec_[slot_idx];
              if (feasign_v[j] != 0) {
                this->thread_dim_keys_[i][shard_id][dim_id].insert(
                    feasign_v[j]);
              }
            }
          }
        }
      };
      for (int i = 0; i < thread_keys_thread_num_; i++) {
        threads.push_back(
            std::thread(gen_dynamic_mf_func,
                        std::ref(vec_data),
                        begin,
                        begin + len_per_thread + (i < remain ? 1 : 0),
                        i));

        begin += len_per_thread + (i < remain ? 1 : 0);
      }
      for (std::thread& t : threads) {
        t.join();
      }
      timeline.Pause();
      VLOG(0) << "GpuPs build task cost " << timeline.ElapsedSec()
              << " seconds.";
    } else {
      PADDLE_ENFORCE_NE(
          data_set_name.find("MultiSlotDataset"),
          std::string::npos,
          common::errors::InvalidArgument(
              "The name of Data set '%s' can not contain 'MultiSlotDataset'.",
              data_set_name.c_str()));
      VLOG(0) << "ps_gpu_wrapper use MultiSlotDataset";
      MultiSlotDataset* dataset = reinterpret_cast<MultiSlotDataset*>(dataset_);
      auto input_channel = dataset->GetInputChannel();

      const std::deque<Record>& vec_data = input_channel->GetData();
      total_len = vec_data.size();
      len_per_thread = total_len / thread_keys_thread_num_;
      remain = total_len % thread_keys_thread_num_;
      auto gen_func = [this](const std::deque<Record>& total_data,
                             int begin_index,
                             int end_index,
                             int i) {
        for (auto iter = total_data.begin() + begin_index;
             iter != total_data.begin() + end_index;
             iter++) {
          const auto& ins = *iter;
          const auto& feasign_v = ins.uint64_feasigns_;
          for (const auto feasign : feasign_v) {
            uint64_t cur_key = feasign.sign().uint64_feasign_;
            int shard_id = cur_key % thread_keys_shard_num_;
            this->thread_keys_[i][shard_id].insert(cur_key);
          }
        }
      };
      for (int i = 0; i < thread_keys_thread_num_; i++) {
        threads.push_back(
            std::thread(gen_func,
                        std::ref(vec_data),
                        begin,
                        begin + len_per_thread + (i < remain ? 1 : 0),
                        i));
        begin += len_per_thread + (i < remain ? 1 : 0);
      }
      for (std::thread& t : threads) {
        t.join();
      }
      timeline.Pause();
      VLOG(0) << "GpuPs build task cost " << timeline.ElapsedSec()
              << " seconds.";
    }
  } else {
    auto& vec_data = dataset_for_pull->GetGpuGraphTotalKeys();
    timeline.Start();
    add_key_to_local(vec_data);
    timeline.Pause();
    VLOG(0) << "GpuGraphTotalKeys: " << vec_data.size()
            << ", add_key_to_local cost " << timeline.ElapsedSec()
            << " seconds.";

    if (FLAGS_graph_edges_split_mode == "fennel" ||
        FLAGS_query_dest_rank_by_multi_node) {
      timeline.Start();
      gpu_task->keys2rank_tables_ = dataset_for_pull->GetPassKeys2RankTable();
      const auto& keys_vec = dataset_for_pull->GetPassKeysVec();
      const auto& ranks_vec = dataset_for_pull->GetPassRanksVec();
      size_t total_keys = 0;
      size_t pred_keys = 0;
      for (auto& keys : keys_vec) {
        total_keys += keys->size();
      }
      pred_keys = total_keys / thread_keys_thread_num_;
      gpu_task->keys2rank_map_vec_.resize(thread_keys_thread_num_);

      auto build_keys2rank_func =
          [this](const std::vector<std::vector<uint64_t>*>& keys_vec,
                 const std::vector<std::vector<uint32_t>*>& ranks_vec,
                 std::vector<std::unordered_map<uint64_t, uint32_t>>&
                     keys2rank_maps,
                 size_t pred_size,
                 int id) {
            keys2rank_maps[id].reserve(pred_size);
            for (size_t i = 0; i < keys_vec.size(); ++i) {
              if (!infer_mode_ || sage_mode_) {
                PADDLE_ENFORCE_EQ(
                    keys_vec[i]->size(),
                    ranks_vec[i]->size(),
                    common::errors::InvalidArgument(
                        "keys_vec[i]->size() should be equal to "
                        "ranks_vec[i]->size(), but received "
                        "keys_vec[i]->size() is %d, ranks_vec[i]->size() is %d",
                        keys_vec[i]->size(),
                        ranks_vec[i]->size()));
                for (size_t j = 0; j < keys_vec[i]->size(); ++j) {
                  auto& key = (*keys_vec[i])[j];
                  auto& rank = (*ranks_vec[i])[j];
                  int shard_idx = key % thread_keys_thread_num_;
                  if (shard_idx == id) {
                    keys2rank_maps[id][key] = rank;
                  }
                }
              } else {
                PADDLE_ENFORCE_EQ(
                    ranks_vec[i]->size(),
                    0UL,
                    common::errors::InvalidArgument(
                        "ranks_vec[i]->size() should be equal to 0, "
                        "but received %d.",
                        ranks_vec[i]->size()));
                for (size_t j = 0; j < keys_vec[i]->size(); ++j) {
                  auto& key = (*keys_vec[i])[j];
                  int shard_idx = key % thread_keys_thread_num_;
                  if (shard_idx == id) {
                    keys2rank_maps[id][key] = rank_id_;
                  }
                }
              }
            }
            VLOG(2) << "build keys2rank_map, shard_idx=" << id
                    << " shard_keys=" << keys2rank_maps[id].size();
          };

      for (int i = 0; i < thread_keys_thread_num_; i++) {
        threads.push_back(std::thread(build_keys2rank_func,
                                      std::ref(keys_vec),
                                      std::ref(ranks_vec),
                                      std::ref(gpu_task->keys2rank_map_vec_),
                                      pred_keys,
                                      i));
      }
      for (std::thread& t : threads) {
        t.join();
      }

      size_t total_shard_keys = 0;
      for (auto& shard : gpu_task->keys2rank_map_vec_) {
        total_shard_keys += shard.size();
      }
      timeline.Pause();
      VLOG(0) << "build keys2rank_map, cost " << timeline.ElapsedSec()
              << " seconds"
              << ", total input keys=" << total_keys
              << ", total uniq keys=" << total_shard_keys;
      PADDLE_ENFORCE_LE(
          total_shard_keys,
          total_keys,
          common::errors::InvalidArgument(
              "Total shard keys number should be less than or equal to total "
              "keys number, but received %d as total shard keys number and %d "
              "as total keys number.",
              total_shard_keys,
              total_keys));
    }
  }

  timeline.Start();
  add_key_to_gputask(gpu_task);
  timeline.Pause();
  VLOG(0) << "add_key_to_gputask cost" << timeline.ElapsedSec() << " seconds.";
}

void PSGPUWrapper::add_slot_feature(std::shared_ptr<HeterContext> gpu_task) {
  platform::Timer timeline;
  platform::Timer time_stage;
  timeline.Start();
  // 8卡数据分片
  size_t device_num = heter_devices_.size();
  std::vector<std::thread> threads;
  auto& local_dim_keys = gpu_task->feature_dim_keys_;  // [shard_num, 0, keys]]
  double divide_nodeid_cost = 0;
  double get_feature_id_cost = 0;
  double add_feature_to_set_cost = 0;
  double add_feature_to_key_cost = 0;

  std::vector<std::vector<uint64_t>> node_ids(device_num);
  size_t node_num = 0;
  for (int i = 0; i < thread_keys_shard_num_; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      node_num += local_dim_keys[i][j].size();
    }
  }
  for (auto& node_id_vector : node_ids) {
    node_id_vector.reserve(node_num * 1.2 / device_num);
  }

  auto& device_dim_mutex = gpu_task->dim_mutex_;

  auto divide_nodeid_to_device =
      [this, device_num, &local_dim_keys, &node_ids, &device_dim_mutex](int i,
                                                                        int j) {
        std::vector<std::vector<uint64_t>> task_keys(device_num);
        size_t batch = 10000;
        for (size_t k = 0; k < device_num; k++) {
          task_keys[k].reserve(batch * 1.2 / device_num);
        }
        std::vector<int> shuffle_device = shuffle_int_vector(device_num);
        size_t start = 0;
        while (start < local_dim_keys[i][j].size()) {
          if (batch + start > local_dim_keys[i][j].size()) {
            batch = local_dim_keys[i][j].size() - start;
          }
          for (size_t k = start; k < (start + batch); k++) {
            int shard = local_dim_keys[i][j][k] % device_num;
            task_keys[shard].push_back(local_dim_keys[i][j][k]);
          }
          // allocate local keys to devices
          for (auto dev : shuffle_device) {
            device_dim_mutex[dev][0]->lock();
            int len = task_keys[dev].size();
            for (int k = 0; k < len; ++k) {
              node_ids[dev].push_back(task_keys[dev][k]);
            }
            device_dim_mutex[dev][0]->unlock();
            task_keys[dev].clear();
          }
          start += batch;
        }
      };
  threads.resize(thread_keys_shard_num_ * multi_mf_dim_);
  time_stage.Start();

  for (int i = 0; i < thread_keys_shard_num_; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      threads[i * multi_mf_dim_ + j] =
          std::thread(divide_nodeid_to_device, i, j);
    }
  }
  for (std::thread& t : threads) {
    t.join();
  }
  threads.clear();
  time_stage.Pause();
  divide_nodeid_cost = time_stage.ElapsedSec();
  if (slot_num_for_pull_feature_ > 0) {
    std::vector<std::vector<uint64_t>> feature_ids(device_num);
    std::vector<uint64_t*> feature_list(device_num);
    std::vector<size_t> feature_list_size(device_num);
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
    if (gpu_graph_mode_) {
      size_t batch = 40000;
      size_t slot_num = static_cast<size_t>(
          slot_num_for_pull_feature_);  // node slot 9008 in slot_vector
      time_stage.Start();
      if (FLAGS_gpugraph_storage_mode ==
          paddle::framework::GpuGraphStorageMode::MEM_EMB_AND_GPU_GRAPH) {
        auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
        auto h_slot_feature_num_map = gpu_graph_ptr->slot_feature_num_map();
        int fea_num_per_node = 0;
        for (size_t i = 0; i < slot_num; ++i) {
          fea_num_per_node += h_slot_feature_num_map[i];
        }

        auto get_feature_id = [this,
                               slot_num,
                               batch,
                               fea_num_per_node,
                               &h_slot_feature_num_map,
                               &node_ids,
                               &feature_ids](int i) {
          platform::CUDADeviceGuard guard(resource_->dev_id(i));
          auto stream = resource_->local_stream(i, 0);
          int* d_slot_feature_num_map;
          uint64_t* d_node_list_ptr;
          uint64_t* d_feature_list_ptr;
          CUDA_CHECK(
              cudaMalloc(reinterpret_cast<void**>(&d_slot_feature_num_map),
                         slot_num * sizeof(int)));
          CUDA_CHECK(cudaMemcpyAsync(d_slot_feature_num_map,
                                     h_slot_feature_num_map.data(),
                                     sizeof(int) * slot_num,
                                     cudaMemcpyHostToDevice,
                                     stream));
          PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_list_ptr),
                                batch * sizeof(uint64_t)));
          CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_feature_list_ptr),
                                batch * fea_num_per_node * sizeof(uint64_t)));
          auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
          uint64_t pos = 0;
          size_t real_batch = 0;
          feature_ids[i].resize(node_ids[i].size() * fea_num_per_node);
          while (pos < node_ids[i].size()) {
            real_batch = (pos + batch) <= node_ids[i].size()
                             ? batch
                             : node_ids[i].size() - pos;
            CUDA_CHECK(cudaMemcpyAsync(d_node_list_ptr,
                                       node_ids[i].data() + pos,
                                       real_batch * sizeof(uint64_t),
                                       cudaMemcpyHostToDevice,
                                       stream));
            int ret =
                gpu_graph_ptr->get_feature_of_nodes(i,
                                                    d_node_list_ptr,
                                                    d_feature_list_ptr,
                                                    real_batch,
                                                    slot_num,
                                                    d_slot_feature_num_map,
                                                    fea_num_per_node);
            PADDLE_ENFORCE_EQ(ret,
                              0,
                              common::errors::PreconditionNotMet(
                                  "Get_feature_of_nodes error."));

            CUDA_CHECK(cudaMemcpyAsync(
                feature_ids[i].data() + pos * fea_num_per_node,
                d_feature_list_ptr,
                real_batch * fea_num_per_node * sizeof(uint64_t),
                cudaMemcpyDeviceToHost,
                stream));
            pos += real_batch;
          }
          PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
          cudaFree(d_slot_feature_num_map);
          cudaFree(d_node_list_ptr);
          cudaFree(d_feature_list_ptr);
        };

        threads.resize(device_num);
        for (size_t i = 0; i < device_num; i++) {
          threads[i] = std::thread(get_feature_id, i);
        }
        for (std::thread& t : threads) {
          t.join();
        }
        threads.clear();
        for (size_t i = 0; i < device_num; i++) {
          feature_list[i] = feature_ids[i].data();
          feature_list_size[i] = feature_ids[i].size();
        }
      } else if (FLAGS_gpugraph_storage_mode ==
                     paddle::framework::GpuGraphStorageMode::
                         MEM_EMB_FEATURE_AND_GPU_GRAPH ||
                 FLAGS_gpugraph_storage_mode ==
                     paddle::framework::GpuGraphStorageMode::
                         SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH) {
        auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
        auto sub_graph_feas =
            gpu_graph_ptr->get_sub_graph_fea(node_ids, slot_num);
        for (size_t i = 0; i < device_num; i++) {
          feature_list[i] = sub_graph_feas[i].feature_list;
          feature_list_size[i] = sub_graph_feas[i].feature_size;
        }
      } else {
        VLOG(0) << "FLAGS_gpugraph_storage_mode is not adaptived";
      }
    }
    time_stage.Pause();
    get_feature_id_cost = time_stage.ElapsedSec();
#endif
    size_t feature_num = 0;
    for (size_t i = 0; i < device_num; i++) {
      feature_num += feature_list_size[i];
    }
    VLOG(1) << "feature_num is " << feature_num << " node_num is " << node_num;

    size_t set_num = thread_keys_shard_num_;
    std::vector<std::unordered_set<uint64_t>> feature_id_set(set_num);
    std::vector<std::mutex> set_mutex(set_num);

    auto add_feature_to_set =
        [this, set_num, &feature_list, &feature_id_set, &set_mutex](
            int dev, size_t start, size_t end) {
          size_t batch = 10000 * set_num;
          std::vector<std::vector<uint64_t>> feature_list_tmp(set_num);
          for (size_t i = 0; i < set_num; i++) {
            feature_list_tmp[i].reserve((batch * 1.2) / set_num);
          }
          std::vector<int> shuffle_set_index = shuffle_int_vector(set_num);
          size_t pos = start;
          size_t real_batch = 0;
          while (pos < end) {
            real_batch = (pos + batch <= end) ? batch : end - pos;
            for (size_t i = pos; i < pos + real_batch; i++) {
              if (feature_list[dev][i] == 0) {
                continue;
              }
              int shard_num = feature_list[dev][i] % set_num;
              feature_list_tmp[shard_num].push_back(feature_list[dev][i]);
            }
            // uniq in local
            for (size_t i = 0; i < set_num; i++) {
              std::sort(feature_list_tmp[i].begin(), feature_list_tmp[i].end());
              size_t idx = 0;
              size_t total = feature_list_tmp[i].size();
              for (size_t j = 0; j < total; j++) {
                auto& k = feature_list_tmp[i][j];
                if (idx > 0 && feature_list_tmp[i][idx - 1] == k) {
                  continue;
                }
                feature_list_tmp[i][idx] = k;
                ++idx;
              }
              feature_list_tmp[i].resize(idx);
            }
            // uniq in global
            for (auto set_index : shuffle_set_index) {
              set_mutex[set_index].lock();
              for (auto feature_id : feature_list_tmp[set_index]) {
                feature_id_set[set_index].insert(feature_id);
              }
              set_mutex[set_index].unlock();
              feature_list_tmp[set_index].clear();
            }
            pos += real_batch;
          }
        };
    size_t device_thread_num = 8;
    threads.resize(device_num * device_thread_num);
    time_stage.Start();
    for (size_t i = 0; i < device_num; i++) {
      size_t start = 0;
      for (size_t j = 0; j < device_thread_num; j++) {
        size_t batch = feature_list_size[i] / device_thread_num;
        if (j < feature_list_size[i] % device_thread_num) {
          batch += 1;
        }
        threads[i * device_thread_num + j] =
            std::thread(add_feature_to_set, i, start, start + batch);
        start += batch;
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
    threads.clear();
    time_stage.Pause();
    add_feature_to_set_cost = time_stage.ElapsedSec();
    auto add_feature_to_key = [this,
                               device_num,
                               &feature_id_set,
                               &local_dim_keys,
                               set_num](int shard_num, int j) {
      local_dim_keys[shard_num][j].reserve(local_dim_keys[shard_num][j].size() +
                                           feature_id_set[shard_num].size());
      for (auto it = feature_id_set[shard_num].begin();
           it != feature_id_set[shard_num].end();
           it++) {
        local_dim_keys[shard_num][j].push_back(*it);
      }
      feature_id_set[shard_num].clear();
    };
    time_stage.Start();
    threads.resize(thread_keys_shard_num_ * multi_mf_dim_);
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        threads[i * multi_mf_dim_ + j] = std::thread(add_feature_to_key, i, j);
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
    time_stage.Pause();
    add_feature_to_key_cost = time_stage.ElapsedSec();
    threads.clear();
    timeline.Pause();
    VLOG(1) << " add_slot_feature costs: " << timeline.ElapsedSec() << " s."
            << " divide_nodeid_cost " << divide_nodeid_cost
            << " get_feature_id_cost " << get_feature_id_cost
            << " add_feature_to_set_cost " << add_feature_to_set_cost
            << " add_feature_to_key_cost " << add_feature_to_key_cost;
  }
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  if (gpu_graph_mode_ && float_slot_num_ > 0) {
    if (FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                           MEM_EMB_FEATURE_AND_GPU_GRAPH ||
        FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                           SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH) {
      auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
      gpu_task->sub_graph_float_feas =
          reinterpret_cast<void*>(new std::vector<GpuPsCommGraphFloatFea>);
      std::vector<GpuPsCommGraphFloatFea>& sub_graph_float_feas = *(
          (std::vector<GpuPsCommGraphFloatFea>*)gpu_task->sub_graph_float_feas);
      if (float_slot_num_ > 0) {
        sub_graph_float_feas =
            gpu_graph_ptr->get_sub_graph_float_fea(node_ids, float_slot_num_);
      }
    }
  }
#endif
}

void PSGPUWrapper::BuildPull(std::shared_ptr<HeterContext> gpu_task) {
  platform::Timer timeline;
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  if (gpu_graph_mode_ &&
      (slot_num_for_pull_feature_ > 0 || float_slot_num_ > 0) &&
      FLAGS_gpugraph_storage_mode !=
          paddle::framework::GpuGraphStorageMode::WHOLE_HBM) {
    add_slot_feature(gpu_task);
  }
#endif
#ifdef PADDLE_WITH_PSLIB
  add_slot_feature(gpu_task);
#endif
  resize_gputask(gpu_task);

  platform::Timer time_stage;
  time_stage.Start();
  gpu_task->UniqueKeys();
  time_stage.Pause();
  VLOG(0) << "passid=" << gpu_task->pass_id_
          << ", BuildPull slot feature uniq and sort cost time: "
          << time_stage.ElapsedSec();

  auto& local_dim_keys = gpu_task->feature_dim_keys_;
  auto& local_dim_ptr = gpu_task->value_dim_ptr_;

  auto& device_dim_keys = gpu_task->device_dim_keys_;
  auto& device_dim_ptr = gpu_task->device_dim_ptr_;

  for (size_t dev = 0; dev < device_dim_keys.size(); dev++) {
    device_dim_keys[dev].resize(multi_mf_dim_);
    device_dim_ptr[dev].resize(multi_mf_dim_);
  }

  // auto& device_mutex = gpu_task->mutex_;

  std::vector<std::thread> threads(thread_keys_shard_num_);

#if (defined PADDLE_WITH_PSLIB) && (defined PADDLE_WITH_HETERPS)
  // get day_id: day nums from 1970
  struct std::tm b;
  b.tm_year = year_ - 1900;
  b.tm_mon = month_ - 1;
  b.tm_mday = day_;
  b.tm_min = b.tm_hour = b.tm_sec = 0;
  std::time_t seconds_from_1970 = std::mktime(&b);
  int day_id = seconds_from_1970 / 86400;
  fleet_ptr_->pslib_ptr_->_worker_ptr->set_day_id(table_id_, day_id);
#endif

  timeline.Start();

  auto ptl_dynamic_mf_func = [this, &local_dim_keys, &local_dim_ptr, &gpu_task](
                                 int i, int j) {
    size_t key_size = local_dim_keys[i][j].size();
    int32_t status = -1;
    int32_t cnt = 0;
#ifdef PADDLE_WITH_PSLIB
    while (true) {
      auto tt = fleet_ptr_->pslib_ptr_->_worker_ptr->pull_sparse_ptr(
          i,
          reinterpret_cast<char**>(local_dim_ptr[i][j].data()),
          this->table_id_,
          local_dim_keys[i][j].data(),
          key_size,
          gpu_task->pass_id_);
      bool flag = true;

      tt.wait();

      try {
        status = tt.get();
      } catch (const std::future_error& e) {
        VLOG(0) << "Caught a future_error with code" << e.code()
                << ", Message:" << e.what();
      }
      if (status != 0) {
        VLOG(0) << "fleet pull sparse failed, status[" << status << "]";
        sleep(sleep_seconds_before_fail_exit_);
        flag = false;
        cnt++;
      }
      if (cnt > 3) {
        VLOG(0) << "fleet pull sparse failed, retry 3 times";
        exit(-1);
      }

      if (flag) {
        break;
      }
    }
#endif
#ifdef PADDLE_WITH_PSCORE
    while (true) {
      auto tt = fleet_ptr_->worker_ptr_->PullSparsePtr(
          i,
          reinterpret_cast<char**>(local_dim_ptr[i][j].data()),
          this->table_id_,
          local_dim_keys[i][j].data(),
          key_size,
          gpu_task->pass_id_,
          gpu_task->keys2rank_map_vec_,
          j);
      bool flag = true;

      tt.wait();

      try {
        status = tt.get();
      } catch (const std::future_error& e) {
        VLOG(0) << "Caught a future_error with code" << e.code()
                << ", Message:" << e.what();
      }
      if (status != 0) {
        VLOG(0) << "fleet pull sparse failed, status[" << status << "]";
        sleep(sleep_seconds_before_fail_exit_);
        flag = false;
        cnt++;
      }
      if (cnt > 3) {
        VLOG(0) << "fleet pull sparse failed, retry 3 times";
        exit(-1);
      }

      if (flag) {
        break;
      }
    }
#endif
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(300);
      exit(-1);
    } else {
      VLOG(2) << "FleetWrapper Pull sparse to local done with table size: "
              << local_dim_keys[i][j].size();
    }
    if (multi_node_) {
      // filter rank data
      FilterPull(gpu_task, i, j);
    }
  };

  threads.resize(thread_keys_shard_num_ * multi_mf_dim_);

  uint64_t total_key = 0;
  std::vector<std::future<void>> task_futures;
  for (int i = 0; i < thread_keys_shard_num_; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      task_futures.emplace_back(
          pull_thread_pool_[i]->enqueue(ptl_dynamic_mf_func, i, j));
      total_key += local_dim_keys[i][j].size();
    }
  }
  for (auto& f : task_futures) {
    f.wait();
  }
  task_futures.clear();
  timeline.Pause();
  VLOG(0) << "passid=" << gpu_task->pass_id_
          << ", pull sparse from CpuPS into GpuPS total keys " << total_key
          << ", cost " << timeline.ElapsedSec() << " seconds.";
}

void PSGPUWrapper::PartitionKey(std::shared_ptr<HeterContext> gpu_task) {
  if (!multi_node_ || !gpu_graph_mode_) {
    return;
  }
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  platform::Timer timeline;

  auto& local_dim_keys = gpu_task->feature_dim_keys_;
  std::vector<std::thread> threads(thread_keys_shard_num_);
  timeline.Start();

  auto re_partition_key_func = [this, &local_dim_keys, &gpu_task](int i,
                                                                  int j) {
    size_t key_size = local_dim_keys[i][j].size();
    int32_t status = -1;
    int32_t cnt = 0;
#ifdef PADDLE_WITH_PSCORE
    while (true) {
      auto tt =
          fleet_ptr_->worker_ptr_->PullSparseKey(i,
                                                 this->table_id_,
                                                 local_dim_keys[i][j].data(),
                                                 key_size,
                                                 gpu_task->pass_id_,
                                                 gpu_task->keys2rank_map_vec_,
                                                 j);
      bool flag = true;

      tt.wait();

      try {
        status = tt.get();
      } catch (const std::future_error& e) {
        VLOG(0) << "Caught a future_error with code" << e.code()
                << ", Message:" << e.what();
      }
      if (status != 0) {
        VLOG(0) << "fleet PartitionKey failed, status[" << status << "]";
        sleep(sleep_seconds_before_fail_exit_);
        flag = false;
        cnt++;
      }
      if (cnt > 3) {
        VLOG(0) << "fleet PartitionKey failed, retry 3 times";
        exit(-1);
      }

      if (flag) {
        break;
      }
    }
#endif
    if (status != 0) {
      LOG(ERROR) << "fleet PartitionKey failed, status[" << status << "]";
      sleep(300);
      exit(-1);
    } else {
      VLOG(2) << "FleetWrapper PartitionKey done with table size: "
              << local_dim_keys[i][j].size();
    }
    if (multi_node_) {
      // filter rank key
      FilterKey(gpu_task, i, j);
    }
  };

  threads.resize(thread_keys_shard_num_ * multi_mf_dim_);

  uint64_t total_key = 0;
  std::vector<std::future<void>> task_futures;
  for (int i = 0; i < thread_keys_shard_num_; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      task_futures.emplace_back(
          pull_thread_pool_[i]->enqueue(re_partition_key_func, i, j));
      total_key += local_dim_keys[i][j].size();
    }
  }
  for (auto& f : task_futures) {
    f.wait();
  }
  task_futures.clear();
  VLOG(0) << "passid=" << gpu_task->pass_id_
          << ", PartitionKey  ori total keys " << total_key;
  MergeKeys(gpu_task);
  timeline.Pause();
  VLOG(0) << "passid=" << gpu_task->pass_id_ << ", PartitionKey "
          << ", cost " << timeline.ElapsedSec() << " seconds.";
#endif
}

void PSGPUWrapper::FilterPull(std::shared_ptr<HeterContext> gpu_task,
                              const int shard_id,
                              const int dim_id) {
#ifdef PADDLE_WITH_HETERPS
  if (!gpu_graph_mode_) {
    return;
  }
  auto& shard_keys = gpu_task->feature_dim_keys_[shard_id][dim_id];
  auto& shard_values = gpu_task->value_dim_ptr_[shard_id][dim_id];
  auto& keys2rank_vec = gpu_task->keys2rank_map_vec_;
  size_t dedup_size = 0;

  for (size_t pos = 0; pos < shard_keys.size(); ++pos) {
    auto& key = shard_keys[pos];
    auto shard_num = keys2rank_vec.size();
    if (shard_num > 0) {
      auto shard = key % shard_num;
      auto it = keys2rank_vec[shard].find(key);
      PADDLE_ENFORCE_NE(
          it,
          keys2rank_vec[shard].end(),
          common::errors::InvalidArgument(
              "Can not find the key in the shard of keys to rank vector."));
      if (static_cast<int>(it->second) != rank_id_) {
        continue;
      }
    } else {
      if (PartitionKeyForRank(key) != rank_id_) {
        continue;
      }
    }
    if (dedup_size == pos) {
      PADDLE_ENFORCE_NE(shard_values[dedup_size],
                        nullptr,
                        common::errors::InvalidArgument(
                            "The shard values after deduplication should not "
                            "be nullptr, but got %d at position %d.",
                            shard_values[dedup_size],
                            dedup_size));
      ++dedup_size;
      continue;
    }
    shard_keys[dedup_size] = shard_keys[pos];
    PADDLE_ENFORCE_NE(shard_values[dedup_size],
                      nullptr,
                      common::errors::InvalidArgument(
                          "The shard values after deduplication should not "
                          "be nullptr, but got %d at position %d.",
                          shard_values[dedup_size],
                          dedup_size));
    ++dedup_size;
  }
  shard_keys.resize(dedup_size);
  shard_values.resize(dedup_size);
#endif
}

void PSGPUWrapper::FilterKey(std::shared_ptr<HeterContext> gpu_task,
                             const int shard_id,
                             const int dim_id) {
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  if (!gpu_graph_mode_) {
    return;
  }
  auto& shard_keys = gpu_task->feature_dim_keys_[shard_id][dim_id];
  auto& keys2rank_vec = gpu_task->keys2rank_map_vec_;
  size_t dedup_size = 0;

  for (size_t pos = 0; pos < shard_keys.size(); ++pos) {
    auto& key = shard_keys[pos];
    auto shard_num = keys2rank_vec.size();
    if (shard_num > 0) {
      auto shard = key % shard_num;
      auto it = keys2rank_vec[shard].find(key);
      PADDLE_ENFORCE_NE(
          it,
          keys2rank_vec[shard].end(),
          common::errors::InvalidArgument(
              "Can not find the key in the shard of keys to rank vector."));
      if (static_cast<int>(it->second) != rank_id_) {
        continue;
      }
    } else {
      if (PartitionKeyForRank(key) != rank_id_) {
        continue;
      }
    }
    if (dedup_size == pos) {
      ++dedup_size;
      continue;
    }
    shard_keys[dedup_size] = shard_keys[pos];
    ++dedup_size;
  }
  shard_keys.resize(dedup_size);
#endif
}

void PSGPUWrapper::MergePull(std::shared_ptr<HeterContext> gpu_task) {
  if (!multi_node_ || !gpu_graph_mode_) {
    return;
  }
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  platform::Timer timeline;
  timeline.Start();
  // barrier
  auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
  if (!gloo_wrapper->IsInitialized()) {
    VLOG(0) << "GLOO is not inited";
    gloo_wrapper->Init();
  }
  gloo_wrapper->Barrier();
  timeline.Pause();

  auto barrier_span = timeline.ElapsedSec();

  timeline.Start();
  std::vector<std::future<void>> task_futures;
  std::vector<std::shared_ptr<paddle::distributed::SparseShardValues>>
      dim_pass_values(multi_mf_dim_, nullptr);
  for (int dim_id = 0; dim_id < multi_mf_dim_; ++dim_id) {
    auto pass_values = fleet_ptr_->worker_ptr_->TakePassSparseReferredValues(
        table_id_, gpu_task->pass_id_, dim_id);
    if (pass_values == nullptr) {
      continue;
    }
    dim_pass_values[dim_id] = pass_values;
    for (int shard_id = 0; shard_id < thread_keys_shard_num_; ++shard_id) {
      auto& merge_values = pass_values->at(shard_id);
      task_futures.emplace_back(pull_thread_pool_[shard_id]->enqueue(
          [this, &gpu_task, &merge_values](int shard_id, int dim_id) {
            auto& shard_keys = gpu_task->feature_dim_keys_[shard_id][dim_id];
            auto& shard_values = gpu_task->value_dim_ptr_[shard_id][dim_id];

            size_t dedup_size = shard_keys.size();
            size_t merge_num = merge_values.keys.size();
            size_t total = merge_num + dedup_size;
            shard_keys.resize(total);
            shard_values.resize(total);

            size_t dedup_index = dedup_size;
            uint64_t last_key = 0;
            if (total > 0) {
              last_key = shard_keys[0];
            }
            size_t i = 0;
            size_t k = 0;

            int num_ranks = node_size_ - 1;
            if (num_ranks == 1) {
              while (i < dedup_size && k < merge_num) {
                auto& merge_key = merge_values.keys[k];
                auto& key = shard_keys[i];
                if ((key == merge_key) || (last_key == merge_key)) {
                  ++k;
                  continue;
                }
                if (key < merge_key) {
                  ++i;
                  continue;
                }
                last_key = merge_key;
                shard_keys[dedup_index] = merge_key;
                shard_values[dedup_index] =
                    CONV2FEATURE_PTR(merge_values.values[k]);
                ++k;
                ++dedup_index;
              }
              if (dedup_size >= 1) {
                uint64_t& key = shard_keys[dedup_size - 1];
                while (k < merge_num) {
                  auto& merge_key = merge_values.keys[k];
                  if (key == merge_key || last_key == merge_key) {
                    ++k;
                    continue;
                  }
                  last_key = merge_key;
                  shard_keys[dedup_index] = merge_key;
                  shard_values[dedup_index] =
                      CONV2FEATURE_PTR(merge_values.values[k]);
                  ++k;
                  ++dedup_index;
                }
              }
            } else {
              merge_values.offsets.push_back(merge_num);
              PADDLE_ENFORCE_EQ(
                  merge_values.offsets.size(),
                  static_cast<size_t>(node_size_),
                  common::errors::InvalidArgument(
                      "Size dismatch in merge_values.offsets.size()(size = "
                      "%zu) and static_cast<size_t>(node_size_)(size = %zu).",
                      merge_values.offsets.size(),
                      static_cast<size_t>(node_size_)));
              std::vector<size_t> ranks_pos(num_ranks);
              for (int rank = 0; rank < num_ranks; ++rank) {
                ranks_pos[rank] = merge_values.offsets[rank];
              }
              ssize_t pos = -1;
              int sel_rank = -1;
              uint64_t min_key = last_key;
              while (i < dedup_size && k < merge_num) {
                auto& key = shard_keys[i];
                if (key < min_key) {
                  ++i;
                  continue;
                }
                if (pos == -1) {
                  for (int rank = 0; rank < num_ranks; ++rank) {
                    size_t& max = merge_values.offsets[rank + 1];
                    size_t& off = ranks_pos[rank];
                    while (off < max) {
                      auto& mkey = merge_values.keys[off];
                      if (key == mkey || last_key == mkey || min_key == mkey) {
                        ++k;
                        ++off;
                        continue;
                      }
                      if (pos == -1 || min_key > mkey) {
                        min_key = mkey;
                        pos = off;
                        sel_rank = rank;
                      }
                      break;
                    }
                  }
                  if (pos == -1) {
                    PADDLE_ENFORCE_EQ((k == merge_num),
                                      true,
                                      common::errors::InvalidArgument(
                                          "shardid=%d, k=%d, merge_num=%d.",
                                          shard_id,
                                          k,
                                          merge_num));
                    break;
                  }
                  if (key < min_key) {
                    ++i;
                    continue;
                  }
                }
                if (min_key != key) {
                  last_key = merge_values.keys[pos];
                  shard_keys[dedup_index] = last_key;
                  shard_values[dedup_index] =
                      CONV2FEATURE_PTR(merge_values.values[pos]);
                  ++dedup_index;
                }
                pos = -1;
                ++k;
                ++ranks_pos[sel_rank];
              }
              uint64_t& key = shard_keys[dedup_size - 1];
              while (k < merge_num) {
                if (pos == -1) {
                  for (int rank = 0; rank < num_ranks; ++rank) {
                    size_t& max = merge_values.offsets[rank + 1];
                    size_t& off = ranks_pos[rank];
                    while (off < max) {
                      auto& mkey = merge_values.keys[off];
                      if (key == mkey || last_key == mkey || min_key == mkey) {
                        ++k;
                        ++off;
                        continue;
                      }
                      if (pos == -1 || min_key > mkey) {
                        min_key = mkey;
                        pos = off;
                        sel_rank = rank;
                      }
                      break;
                    }
                  }
                  if (pos == -1) {
                    PADDLE_ENFORCE_EQ((k == merge_num),
                                      true,
                                      common::errors::InvalidArgument(
                                          "shardid=%d, k=%d, merge_num=%d.",
                                          shard_id,
                                          k,
                                          merge_num));
                    break;
                  }
                }
                last_key = merge_values.keys[pos];
                shard_keys[dedup_index] = last_key;
                shard_values[dedup_index] =
                    CONV2FEATURE_PTR(merge_values.values[pos]);
                ++dedup_index;
                pos = -1;
                ++k;
                ++ranks_pos[sel_rank];
              }
            }
            shard_keys.resize(dedup_index);
            shard_values.resize(dedup_index);
          },
          shard_id,
          dim_id));
    }
  }
  for (auto& f : task_futures) {
    f.wait();
  }
  task_futures.clear();

  uint64_t total_key = 0;
  for (int shard_id = 0; shard_id < thread_keys_shard_num_; ++shard_id) {
    for (int dim_id = 0; dim_id < multi_mf_dim_; ++dim_id) {
      total_key += gpu_task->feature_dim_keys_[shard_id][dim_id].size();
    }
  }
  timeline.Pause();
  VLOG(0) << "passid=" << gpu_task->pass_id_
          << ", merge pull sparse from CpuPS into GpuPS total keys "
          << total_key << ", cost " << timeline.ElapsedSec()
          << " seconds, barrier span: " << barrier_span;
#endif
}

void PSGPUWrapper::MergeKeys(std::shared_ptr<HeterContext> gpu_task) {
  if (!multi_node_ || !gpu_graph_mode_) {
    return;
  }
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  platform::Timer timeline;
  timeline.Start();
  // barrier
  auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
  if (!gloo_wrapper->IsInitialized()) {
    VLOG(0) << "GLOO is not inited";
    gloo_wrapper->Init();
  }
  gloo_wrapper->Barrier();
  timeline.Pause();

  auto barrier_span = timeline.ElapsedSec();

  timeline.Start();
  std::vector<std::future<void>> task_futures;
  std::vector<std::shared_ptr<paddle::distributed::SparseShardValues>>
      dim_pass_values(multi_mf_dim_, nullptr);
  for (int dim_id = 0; dim_id < multi_mf_dim_; ++dim_id) {
    auto pass_values = fleet_ptr_->worker_ptr_->TakePassSparseReferredValues(
        table_id_, gpu_task->pass_id_, dim_id);
    if (pass_values == nullptr) {
      continue;
    }
    dim_pass_values[dim_id] = pass_values;
    for (int shard_id = 0; shard_id < thread_keys_shard_num_; ++shard_id) {
      auto& merge_values = pass_values->at(shard_id);
      task_futures.emplace_back(pull_thread_pool_[shard_id]->enqueue(
          [this, &gpu_task, &merge_values](int shard_id, int dim_id) {
            auto& shard_keys = gpu_task->feature_dim_keys_[shard_id][dim_id];
            size_t dedup_size = shard_keys.size();
            size_t merge_num = merge_values.keys.size();
            size_t total = merge_num + dedup_size;
            shard_keys.resize(total);
            size_t dedup_index = dedup_size;
            uint64_t last_key = shard_keys[0];

            size_t i = 0;
            size_t k = 0;

            int num_ranks = node_size_ - 1;
            if (num_ranks == 1) {
              while (i < dedup_size && k < merge_num) {
                auto& merge_key = merge_values.keys[k];
                auto& key = shard_keys[i];
                if ((key == merge_key) || (last_key == merge_key)) {
                  ++k;
                  continue;
                }
                if (key < merge_key) {
                  ++i;
                  continue;
                }
                last_key = merge_key;
                shard_keys[dedup_index] = merge_key;
                ++k;
                ++dedup_index;
              }
              uint64_t& key = shard_keys[dedup_size - 1];
              while (k < merge_num) {
                auto& merge_key = merge_values.keys[k];
                if (key == merge_key || last_key == merge_key) {
                  ++k;
                  continue;
                }
                last_key = merge_key;
                shard_keys[dedup_index] = merge_key;
                ++k;
                ++dedup_index;
              }
            } else {
              merge_values.offsets.push_back(merge_num);
              PADDLE_ENFORCE_EQ(
                  merge_values.offsets.size(),
                  static_cast<size_t>(node_size_),
                  common::errors::InvalidArgument(
                      "Size dismatch in merge_values.offsets.size()(size = "
                      "%zu) and static_cast<size_t>(node_size_)(size = %zu).",
                      merge_values.offsets.size(),
                      static_cast<size_t>(node_size_)));
              std::vector<size_t> ranks_pos(num_ranks);
              for (int rank = 0; rank < num_ranks; ++rank) {
                ranks_pos[rank] = merge_values.offsets[rank];
              }
              ssize_t pos = -1;
              int sel_rank = -1;
              uint64_t min_key = last_key;
              while (i < dedup_size && k < merge_num) {
                auto& key = shard_keys[i];
                if (key < min_key) {
                  ++i;
                  continue;
                }
                if (pos == -1) {
                  for (int rank = 0; rank < num_ranks; ++rank) {
                    size_t& max = merge_values.offsets[rank + 1];
                    size_t& off = ranks_pos[rank];
                    while (off < max) {
                      auto& mkey = merge_values.keys[off];
                      if (key == mkey || last_key == mkey || min_key == mkey) {
                        ++k;
                        ++off;
                        continue;
                      }
                      if (pos == -1 || min_key > mkey) {
                        min_key = mkey;
                        pos = off;
                        sel_rank = rank;
                      }
                      break;
                    }
                  }
                  if (pos == -1) {
                    PADDLE_ENFORCE_EQ((k == merge_num),
                                      true,
                                      common::errors::InvalidArgument(
                                          "shardid=%d, k=%d, merge_num=%d.",
                                          shard_id,
                                          k,
                                          merge_num));
                    break;
                  }
                  if (key < min_key) {
                    ++i;
                    continue;
                  }
                }
                if (min_key != key) {
                  last_key = merge_values.keys[pos];
                  shard_keys[dedup_index] = last_key;
                  ++dedup_index;
                }
                pos = -1;
                ++k;
                ++ranks_pos[sel_rank];
              }
              uint64_t& key = shard_keys[dedup_size - 1];
              while (k < merge_num) {
                if (pos == -1) {
                  for (int rank = 0; rank < num_ranks; ++rank) {
                    size_t& max = merge_values.offsets[rank + 1];
                    size_t& off = ranks_pos[rank];
                    while (off < max) {
                      auto& mkey = merge_values.keys[off];
                      if (key == mkey || last_key == mkey || min_key == mkey) {
                        ++k;
                        ++off;
                        continue;
                      }
                      if (pos == -1 || min_key > mkey) {
                        min_key = mkey;
                        pos = off;
                        sel_rank = rank;
                      }
                      break;
                    }
                  }
                  if (pos == -1) {
                    PADDLE_ENFORCE_EQ((k == merge_num),
                                      true,
                                      common::errors::InvalidArgument(
                                          "shardid=%d, k=%d, merge_num=%d.",
                                          shard_id,
                                          k,
                                          merge_num));
                    break;
                  }
                }
                last_key = merge_values.keys[pos];
                shard_keys[dedup_index] = last_key;
                ++dedup_index;
                pos = -1;
                ++k;
                ++ranks_pos[sel_rank];
              }
            }
            shard_keys.resize(dedup_index);
          },
          shard_id,
          dim_id));
    }
  }
  for (auto& f : task_futures) {
    f.wait();
  }
  task_futures.clear();

  uint64_t total_key = 0;
  for (int shard_id = 0; shard_id < thread_keys_shard_num_; ++shard_id) {
    for (int dim_id = 0; dim_id < multi_mf_dim_; ++dim_id) {
      total_key += gpu_task->feature_dim_keys_[shard_id][dim_id].size();
    }
  }
  timeline.Pause();
  VLOG(0) << "passid=" << gpu_task->pass_id_ << ", merge keys " << total_key
          << ", cost " << timeline.ElapsedSec()
          << " seconds, barrier span: " << barrier_span;
#endif
}

void PSGPUWrapper::divide_to_device(std::shared_ptr<HeterContext> gpu_task) {
  platform::Timer timeline;
  int device_num = heter_devices_.size();
  std::vector<std::future<void>> task_futures;
  auto& local_dim_keys = gpu_task->feature_dim_keys_;
  auto& local_dim_ptr = gpu_task->value_dim_ptr_;

  auto& device_dim_keys = gpu_task->device_dim_keys_;
  auto& device_dim_ptr = gpu_task->device_dim_ptr_;
  auto& device_dim_mutex = gpu_task->dim_mutex_;
  // auto& device_mutex = gpu_task->mutex_;

  if (multi_mf_dim_) {
    for (size_t dev = 0; dev < device_dim_keys.size(); dev++) {
      device_dim_keys[dev].resize(multi_mf_dim_);
      device_dim_ptr[dev].resize(multi_mf_dim_);
    }
  }

  timeline.Start();
  auto build_pull_dynamic_mf_func = [this,
                                     device_num,
                                     &local_dim_keys,
                                     &local_dim_ptr,
                                     &device_dim_keys,
                                     &device_dim_ptr,
                                     &device_dim_mutex](int i, int j) {
    thread_local std::vector<std::vector<uint32_t>> task_pos(device_num);
    auto& h_dim_keys = local_dim_keys[i][j];
    size_t total_keys_len = h_dim_keys.size();
    for (int i = 0; i < device_num; ++i) {
      task_pos[i].reserve((total_keys_len + device_num - 1) / device_num);
      task_pos[i].clear();
    }
    for (size_t k = 0; k < total_keys_len; k++) {
      int shard = h_dim_keys[k] % device_num;
      task_pos[shard].push_back(k);
    }
    auto& h_dim_ptrs = local_dim_ptr[i][j];
    // allocate local keys to devices
    std::vector<int> shuffle_device = shuffle_int_vector(device_num);
    for (auto dev : shuffle_device) {
      device_dim_mutex[dev][j]->lock();
      auto& dev_pos = task_pos[dev];
      size_t len = dev_pos.size();
      auto& d_dim_keys = device_dim_keys[dev][j];
      auto& d_dim_ptr = device_dim_ptr[dev][j];
      size_t cur = d_dim_keys.size();
      size_t total = cur + len;
      d_dim_keys.resize(total);
      d_dim_ptr.resize(total);
      for (size_t k = 0; k < len; ++k) {
        auto& pos = dev_pos[k];
        d_dim_keys[cur + k] = h_dim_keys[pos];
        PADDLE_ENFORCE_NE(h_dim_ptrs[pos],
                          nullptr,
                          common::errors::InvalidArgument(
                              "The value of local dimension pointer should not "
                              "be nullptr but received %d at position %d.",
                              h_dim_ptrs[pos],
                              pos));
        d_dim_ptr[cur + k] = h_dim_ptrs[pos];
      }
      device_dim_mutex[dev][j]->unlock();
    }
  };

  if (multi_mf_dim_) {
    task_futures.clear();
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        int tid = (i * multi_mf_dim_ + j) % device_num_;
        task_futures.emplace_back(
            cpu_work_pool_[tid]->enqueue(build_pull_dynamic_mf_func, i, j));
      }
    }
    for (auto& f : task_futures) {
      f.wait();
    }
  }
  timeline.Pause();
  VLOG(1) << "passid=" << gpu_task->pass_id_
          << ", GpuPs prepare for build hbm cost " << timeline.ElapsedSec()
          << " seconds.";
}

void PSGPUWrapper::PrepareGPUTask(std::shared_ptr<HeterContext> gpu_task) {
  platform::Timer timeline;
  int device_num = heter_devices_.size();
  std::vector<std::thread> threads;
  std::vector<std::future<void>> task_futures;
  auto& local_keys = gpu_task->feature_keys_;
  auto& local_ptr = gpu_task->value_ptr_;

  auto& device_keys = gpu_task->device_keys_;
  auto& device_vals = gpu_task->device_values_;
  // auto& device_mutex = gpu_task->mutex_;

  timeline.Start();
  std::vector<std::vector<std::pair<uint64_t, char*>>> pass_values;

  bool record_status = false;
  auto& device_task_keys = gpu_task->device_task_keys_;
  auto& device_task_ptrs = gpu_task->device_task_ptr_;

  auto build_func = [device_num,
                     record_status,
                     &pass_values,
                     &local_keys,
                     &local_ptr,
                     &device_task_keys,
                     &device_task_ptrs](int i) {
    auto& task_keys = device_task_keys[i];
#ifdef PADDLE_WITH_PSLIB
    auto& task_ptrs = device_task_ptrs[i];
#endif

#ifdef PADDLE_WITH_PSCORE
    auto& task_ptrs = device_task_ptrs[i];
#endif

    for (size_t j = 0; j < local_keys[i].size(); j++) {
      int shard = local_keys[i][j] % device_num;
      task_keys[shard].push_back(local_keys[i][j]);
      task_ptrs[shard].push_back(local_ptr[i][j]);
    }
#ifdef PADDLE_WITH_PSLIB
    if (record_status) {
      size_t local_keys_size = local_keys.size();
      size_t pass_values_size = pass_values.size();
      for (size_t j = 0; j < pass_values_size; j += local_keys_size) {
        auto& shard_values = pass_values[j];
        for (size_t pair_idx = 0; pair_idx < pass_values[j].size();
             pair_idx++) {
          auto& cur_pair = shard_values[pair_idx];
          int shard = cur_pair.first % device_num;
          task_keys[shard].push_back(cur_pair.first);
          task_ptrs[shard].push_back(
              (paddle::ps::DownpourFixedFeatureValue*)cur_pair.second);
        }
      }
    }
#endif
  };
  if (!multi_mf_dim_) {
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      int tid = i % device_num_;
      task_futures.emplace_back(cpu_work_pool_[tid]->enqueue(build_func, i));
    }
    for (auto& f : task_futures) {
      f.wait();
    }
    task_futures.clear();
    VLOG(0) << "GpuPs build hbmps done";
  }
  std::vector<std::vector<int>> prefix_sum;
  prefix_sum.resize(device_num);
  for (int i = 0; i < device_num; i++) {
    prefix_sum[i].resize(thread_keys_shard_num_ + 1);
    prefix_sum[i][0] = 0;
  }
  auto calc_prefix_func = [this,
                           &prefix_sum,
                           &device_keys,
                           &device_vals,
                           &device_task_keys](int device_num) {
    for (int j = 0; j < thread_keys_shard_num_; j++) {
      prefix_sum[device_num][j + 1] =
          prefix_sum[device_num][j] + device_task_keys[j][device_num].size();
    }
    device_keys[device_num].resize(
        prefix_sum[device_num][thread_keys_shard_num_]);
    device_vals[device_num].resize(
        prefix_sum[device_num][thread_keys_shard_num_]);
  };
  if (!multi_mf_dim_) {
    for (int i = 0; i < device_num; i++) {
      task_futures.emplace_back(
          hbm_thread_pool_[i]->enqueue(calc_prefix_func, i));
    }
    for (auto& f : task_futures) {
      f.wait();
    }
    task_futures.clear();
  }
  VLOG(0) << "prefix done";
  timeline.Pause();
  VLOG(0) << "passid=" << gpu_task->pass_id_
          << ", GpuPs prepare for build hbm cost " << timeline.ElapsedSec()
          << " seconds.";
}

void PSGPUWrapper::BuildGPUTask(std::shared_ptr<HeterContext> gpu_task) {
  int device_num = heter_devices_.size();
  platform::Timer stagetime;
  stagetime.Start();

  std::vector<size_t> feature_keys_count(device_num);
  size_t size_max = 0;

  for (int i = 0; i < device_num; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      feature_keys_count[i] += gpu_task->device_dim_ptr_[i][j].size();
      VLOG(2) << i << " card with dynamic mf dim: " << index_dim_vec_[j]
              << " dim index: " << j << " contains feasign nums: "
              << gpu_task->device_dim_ptr_[i][j].size();
    }
    VLOG(2) << i << " card with dynamic mf contains feasign nums total: "
            << feature_keys_count[i];
    size_max = std::max(size_max, feature_keys_count[i]);
    if (FLAGS_graph_edges_split_mode == "fennel" ||
        FLAGS_query_dest_rank_by_multi_node) {
      resource_->set_keys2rank(i, gpu_task->keys2rank_tables_[i]);
    }
  }
  if (size_max <= 0) {
    VLOG(0) << "Skip build gpu ps cause feasign nums = " << size_max;
    return;
  }

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  if (HeterPs_ == NULL) {
    HeterPs_ = HeterPsBase::get_instance(
        size_max, resource_, fleet_config_, accessor_class_, optimizer_type_);
#ifdef PADDLE_WITH_CUDA
    HeterPs_->set_nccl_comm_and_size(
        inner_comms_, inter_comms_, node_size_, rank_id_);
    HeterPs_->set_sparse_sgd(optimizer_config_);
    HeterPs_->set_embedx_sgd(optimizer_config_);
#endif
  }
  stagetime.Pause();
  VLOG(1) << "passid=" << gpu_task->pass_id_ << ", card: "
          << " BuildGPUTask create HeterPs_ costs: " << stagetime.ElapsedSec()
          << " s.";
  stagetime.Start();

  auto build_dynamic_mf_func =
      [this, &gpu_task, &accessor_wrapper_ptr](
          const int i, const size_t tid, const size_t once_gpu_copy) {
        //    VLOG(0) << "begin build_dynamic_mf_func tid=" << tid << ", i=" <<
        //    i;
        for (int j = 0; j < multi_mf_dim_; j++) {
          auto& device_dim_ptrs = gpu_task->device_dim_ptr_[i][j];
          int mf_dim = this->index_dim_vec_[j];
          size_t feature_value_size =
              accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);
          size_t len = device_dim_ptrs.size();
          size_t start = tid * once_gpu_copy;
          while (start < len) {
            size_t real_len =
                (len - start) > once_gpu_copy ? once_gpu_copy : (len - start);
            size_t end = start + real_len;
            std::shared_ptr<char> build_values(
                new char[feature_value_size * real_len],
                [](char* p) { delete[] p; });
            char* test_build_values = build_values.get();
            for (size_t k = start; k < end; k++) {
#ifdef PADDLE_WITH_PSCORE
              void* val = reinterpret_cast<float*>(
                  test_build_values + (k - start) * feature_value_size);
              accessor_wrapper_ptr->BuildFill(
                  val, device_dim_ptrs[k], cpu_table_accessor_, mf_dim);
#endif
#ifdef PADDLE_WITH_PSLIB
              float* val = reinterpret_cast<float*>(
                  test_build_values + (k - start) * feature_value_size);
              accessor_wrapper_ptr->BuildFill(val,
                                              device_dim_ptrs[k],
                                              cpu_table_accessor_,
                                              mf_dim,
                                              accessor_class_);
#endif
            }
            task_info task;
            task.build_values = build_values;
            task.offset = start;
            task.device_id = i;
            task.multi_mf_dim = j;
            task.start = 0;
            task.end = static_cast<int>(real_len);
            cpu_reday_channels_[i]->Put(task);
            // step
            start = start + (once_gpu_copy * cpu_device_thread_num_);
          }
        }
        //    VLOG(0) << "end build_dynamic_mf_func tid=" << tid << ", i=" << i;
      };

  auto build_dymf_hbm_pool = [this,
                              &gpu_task,
                              &accessor_wrapper_ptr,
                              &feature_keys_count](int i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(i));

    platform::Timer stagetime;
    platform::Timer timer;
    timer.Start();
    // reset table
    this->HeterPs_->reset_table(i,
                                feature_keys_count[i],
                                optimizer_config_,
                                optimizer_config_,
                                infer_mode_);
    // insert hbm table
    stagetime.Start();
    for (int j = 0; j < multi_mf_dim_; j++) {
      auto& device_dim_keys = gpu_task->device_dim_keys_[i][j];
      size_t len = device_dim_keys.size();
      int mf_dim = this->index_dim_vec_[j];
      size_t feature_value_size =
          accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);
      this->hbm_pools_[i * this->multi_mf_dim_ + j]->reset(len,
                                                           feature_value_size);
      this->HeterPs_->build_ps(
          i,
          device_dim_keys.data(),
          this->hbm_pools_[i * this->multi_mf_dim_ + j]->mem(),
          len,
          feature_value_size,
          4 * 1024 * 1024,
          2);
      if (device_dim_keys.size() > 0) {
        VLOG(3) << "show table: " << i
                << " table kv size: " << device_dim_keys.size()
                << "dim: " << this->index_dim_vec_[j] << " len: " << len;
        HeterPs_->show_one_table(i);
      }
    }

    stagetime.Pause();
    auto build_span = stagetime.ElapsedSec();

    stagetime.Start();
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
    // build feature table
    if (gpu_graph_mode_ && slot_num_for_pull_feature_ > 0 &&
        (FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                            MEM_EMB_FEATURE_AND_GPU_GRAPH ||
         FLAGS_gpugraph_storage_mode ==
             paddle::framework::GpuGraphStorageMode::
                 SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH)) {
      auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
      std::vector<GpuPsCommGraphFea>* tmp =
          (std::vector<GpuPsCommGraphFea>*)gpu_task->sub_graph_feas;
      gpu_graph_ptr->build_gpu_graph_fea((*tmp)[i], i);
    }

    if (gpu_graph_mode_ && float_slot_num_ > 0 &&
        (FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                            MEM_EMB_FEATURE_AND_GPU_GRAPH ||
         FLAGS_gpugraph_storage_mode ==
             paddle::framework::GpuGraphStorageMode::
                 SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH)) {
      auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
      std::vector<GpuPsCommGraphFloatFea>* float_tmp =
          (std::vector<GpuPsCommGraphFloatFea>*)gpu_task->sub_graph_float_feas;
      gpu_graph_ptr->build_gpu_graph_float_fea((*float_tmp)[i], i);
    }
#endif
    stagetime.Pause();
    auto build_feature_span = stagetime.ElapsedSec();

    size_t total_len = 0;
    stagetime.Start();
    struct task_info task;
    auto stream = resource_->local_stream(i, 0);
    while (cpu_reday_channels_[i]->Get(task)) {
      auto hbm = this->hbm_pools_[task.device_id * this->multi_mf_dim_ +
                                  task.multi_mf_dim]
                     ->mem();
      int mf_dim = this->index_dim_vec_[task.multi_mf_dim];
      size_t feature_value_size =
          accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);
      auto hbm_start = hbm + task.offset * feature_value_size;
      CUDA_CHECK(cudaMemcpyAsync(
          hbm_start,
          task.build_values.get() + task.start * feature_value_size,
          (task.end - task.start) * feature_value_size,
          cudaMemcpyHostToDevice,
          stream));
      total_len += (task.end - task.start);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    stagetime.Pause();
    timer.Pause();

    VLOG(1) << "card: " << i
            << " BuildGPUTask build_ps total costs: " << timer.ElapsedSec()
            << ", copy: " << stagetime.ElapsedSec() << ", table: " << build_span
            << ", feature: " << build_feature_span
            << ", feasign: " << total_len;
  };

  std::vector<std::future<void>> cpu_task_futures;
  std::vector<std::future<void>> gpu_task_futures;

  int once_gpu_copy = 64 * 1024;
  for (int i = 0; i < device_num; i++) {
    cpu_reday_channels_[i]->Open();
    gpu_task_futures.emplace_back(
        hbm_thread_pool_[i]->enqueue(build_dymf_hbm_pool, i));
    for (size_t tid = 0; tid < cpu_device_thread_num_; ++tid) {
      cpu_task_futures.emplace_back(cpu_work_pool_[i]->enqueue(
          build_dynamic_mf_func, i, tid, once_gpu_copy));
    }
  }

  stagetime.Start();
  for (auto& f : cpu_task_futures) {
    f.wait();
  }
  cpu_task_futures.clear();
  stagetime.Pause();
  VLOG(1) << "passid=" << gpu_task->pass_id_
          << ", BuildGPUTask build_dynamic_mf_func "
          << " cost " << stagetime.ElapsedSec() << " s.";
  for (int i = 0; i < device_num; i++) {
    cpu_reday_channels_[i]->Close();
  }
  stagetime.Start();
  for (auto& f : gpu_task_futures) {
    f.wait();
  }
  gpu_task_futures.clear();
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  if (gpu_graph_mode_ &&
      (FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                          MEM_EMB_FEATURE_AND_GPU_GRAPH ||
       FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                          SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH)) {
    std::vector<GpuPsCommGraphFea>* tmp =
        (std::vector<GpuPsCommGraphFea>*)gpu_task->sub_graph_feas;
    delete tmp;
    gpu_task->sub_graph_feas = NULL;

    std::vector<GpuPsCommGraphFloatFea>* float_tmp =
        (std::vector<GpuPsCommGraphFloatFea>*)gpu_task->sub_graph_float_feas;
    delete float_tmp;
    gpu_task->sub_graph_float_feas = NULL;
  }
#endif
  stagetime.Pause();
  VLOG(1) << "  build_dymf_hbm_pool "
          << " cost " << stagetime.ElapsedSec() << " s.";
}

void PSGPUWrapper::LoadIntoMemory(bool is_shuffle) {
  platform::Timer timer;
  VLOG(3) << "Begin LoadIntoMemory(), dataset[" << dataset_ << "]";
  timer.Start();
  dataset_->LoadIntoMemory();
  timer.Pause();
  int64_t total_ins_num = dataset_->GetMemoryDataSize();
  VLOG(1) << "LoadIntoMemory cost: " << timer.ElapsedSec()
          << "s, total ins num:" << total_ins_num << ", dataset[" << dataset_
          << "]";
  gpu_graph_mode_ = dataset_->GetGpuGraphMode();
  if (total_ins_num == 0) {
    VLOG(0) << "GetMemoryDataSize == 0";
    return;
  }
  // local shuffle
  if (is_shuffle) {
    dataset_->LocalShuffle();
  }

  InitSlotInfo();
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  if (gpu_graph_mode_) {
    if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
      AddSparseKeys();
    } else if (hbm_sparse_table_initialized_ == false) {
      SparseTableToHbm();
    }
  } else {
    AddSparseKeys();
  }
#else
  AddSparseKeys();
#endif
  VLOG(1) << "End LoadIntoMemory(), dataset[" << dataset_ << "]";
}

void PSGPUWrapper::start_build_thread() {
  running_ = true;
  VLOG(3) << "start build CPU ps thread.";
  buildpull_threads_ = std::thread([this] { build_pull_thread(); });
}

void PSGPUWrapper::AddSparseKeys() {
  std::shared_ptr<HeterContext> gpu_task = gpu_task_pool_.Get();
  gpu_task->Reset();
  gpu_task->pass_id_ = static_cast<uint16_t>(dataset_->GetPassID());

  platform::Timer timer;
  timer.Start();
  // build cpu ps data process
  PreBuildTask(gpu_task, dataset_);
  timer.Pause();
  VLOG(1) << "passid=" << gpu_task->pass_id_
          << ", thread PreBuildTask end, cost time: " << timer.ElapsedSec()
          << " s";
  buildcpu_ready_channel_->Put(gpu_task);
}

void PSGPUWrapper::build_pull_thread() {
  while (running_) {
    std::shared_ptr<HeterContext> gpu_task = nullptr;
    if (!buildcpu_ready_channel_->Get(gpu_task)) {
      continue;
    }
    VLOG(3) << "thread build pull start.";
    platform::Timer timer;
    timer.Start();
    if (slot_num_for_pull_feature_ > 0 || float_slot_num_ > 0) {
      PartitionKey(gpu_task);
    }
    // build cpu ps data process
    BuildPull(gpu_task);
    timer.Pause();
    VLOG(0) << "passid=" << gpu_task->pass_id_
            << ", thread BuildPull end, cost time: " << timer.ElapsedSec()
            << "s";
    buildpull_ready_channel_->Put(gpu_task);
  }
  VLOG(3) << "build cpu thread end";
}

void PSGPUWrapper::build_task() {
  // build_task: build_pull + build_gputask
  std::shared_ptr<HeterContext> gpu_task = nullptr;
  // ins and pre_build end
  if (!buildpull_ready_channel_->Get(gpu_task)) {
    return;
  }

  VLOG(1) << "passid=" << gpu_task->pass_id_ << ", PrepareGPUTask start.";
  platform::Timer timer;
  timer.Start();
  // merge pull
  MergePull(gpu_task);
  if (multi_mf_dim_) {
    divide_to_device(gpu_task);
  } else {
    PrepareGPUTask(gpu_task);
  }
  BuildGPUTask(gpu_task);
  timer.Pause();
  VLOG(1) << "passid=" << gpu_task->pass_id_
          << ", PrepareGPUTask + BuildGPUTask end, cost time: "
          << timer.ElapsedSec() << "s";

  current_task_ = gpu_task;
}

void PSGPUWrapper::BeginPass() {
  platform::Timer timer;
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  if (gpu_graph_mode_ &&
      (FLAGS_gpugraph_storage_mode == GpuGraphStorageMode::WHOLE_HBM)) {
    return;
  }
#endif
  timer.Start();
  if (current_task_) {
    PADDLE_THROW(
        common::errors::Fatal("[BeginPass] current task is not ended."));
  }

  debug_gpu_memory_info("before build task");
  build_task();
  debug_gpu_memory_info("after build task");
  timer.Pause();

  if (current_task_ == nullptr) {
    PADDLE_THROW(common::errors::Fatal(
        "[BeginPass] after build_task, current task is not null."));
  }
  if (FLAGS_gpugraph_dedup_pull_push_mode) {
    VLOG(0) << "passid=" << current_task_->pass_id_
            << ", BeginPass end, cost time: " << timer.ElapsedSec()
            << "s, enable pull push dedup mode="
            << FLAGS_gpugraph_dedup_pull_push_mode;
  } else {
    VLOG(0) << "passid=" << current_task_->pass_id_
            << ", BeginPass end, cost time: " << timer.ElapsedSec() << "s";
  }
}

void PSGPUWrapper::EndPass() {
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  if (gpu_graph_mode_ &&
      (FLAGS_gpugraph_storage_mode == GpuGraphStorageMode::WHOLE_HBM)) {
    return;
  }
#endif
  if (current_task_ == nullptr) {
    return;
  }
  platform::Timer stagetime;
  stagetime.Start();
  HbmToSparseTable();
  stagetime.Pause();
  VLOG(0) << "passid=" << current_task_->pass_id_
          << ", EndPass HbmToSparseTable cost time: " << stagetime.ElapsedSec()
          << "s";

  gpu_task_pool_.Push(current_task_);
  current_task_ = nullptr;
  // fleet_ptr->pslib_ptr_->_worker_ptr->release_table_mutex(this->table_id_);
}

void PSGPUWrapper::SparseTableToHbm() {
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  if (!gpu_graph_mode_) {
    return;
  }
  std::shared_ptr<HeterContext> gpu_task = gpu_task_pool_.Get();
  gpu_task->Reset();
  size_t device_num = heter_devices_.size();
  gpu_task->init(thread_keys_shard_num_, device_num, multi_mf_dim_);
  gpu_task->pass_id_ = (uint16_t)(dataset_->GetPassID());
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto node_to_id = gpu_graph_ptr->node_to_id;
  auto edge_to_id = gpu_graph_ptr->edge_to_id;
  std::vector<uint64_t> vec_data = gpu_graph_ptr->get_graph_total_keys();

  thread_dim_keys_.resize(thread_keys_thread_num_);
  for (int i = 0; i < thread_keys_thread_num_; i++) {
    thread_dim_keys_[i].resize(thread_keys_shard_num_);
    for (int j = 0; j < thread_keys_shard_num_; j++) {
      thread_dim_keys_[i][j].resize(multi_mf_dim_);
    }
  }

  add_key_to_local(vec_data);
  add_key_to_gputask(gpu_task);
  if (slot_num_for_pull_feature_ > 0 || float_slot_num_ > 0) {
    PartitionKey(gpu_task);
  }
  BuildPull(gpu_task);
  MergePull(gpu_task);
  if (!multi_mf_dim_) {
    PrepareGPUTask(gpu_task);
  } else {
    divide_to_device(gpu_task);
  }
  BuildGPUTask(gpu_task);
  current_task_ = gpu_task;
  hbm_sparse_table_initialized_ = true;
#endif
}

void PSGPUWrapper::HbmToSparseTable() {
  // hbm no update not need dump
  if (grad_push_count_ == 0) {
    return;
  }
  grad_push_count_ = 0;

  if (!current_task_) {
    PADDLE_THROW(
        common::errors::Fatal("[EndPass] current task has been ended."));
  }
  size_t keysize_max = 0;
  // in case of feasign_num = 0, skip dump_to_cpu

  for (size_t i = 0; i < heter_devices_.size(); i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      keysize_max =
          std::max(keysize_max, current_task_->device_dim_keys_[i][j].size());
    }
  }
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();

  int once_cpu_num = 16 * 1024;
  int once_gpu_copy = 8 * once_cpu_num;

  auto dump_pool_to_cpu_func = [this, &accessor_wrapper_ptr, once_cpu_num](
                                   int i, size_t once_gpu_copy) {
    platform::Timer tm;
    tm.Start();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(this->resource_->dev_id(i)));
    auto stream = this->resource_->local_stream(i, 0);

    size_t total_len = 0;
    // multi mf dim
    for (int j = 0; j < this->multi_mf_dim_; ++j) {
      auto& hbm_pool = this->hbm_pools_[i * this->multi_mf_dim_ + j];
      // ============ multi-thread process feasign============
      int mf_dim = this->index_dim_vec_[j];
      size_t feature_value_size =
          accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);

      auto& device_keys = this->current_task_->device_dim_keys_[i][j];
      size_t len = device_keys.size();
      size_t start = 0;
      while (start < len) {
        size_t real_len =
            (len - start) >= once_gpu_copy ? once_gpu_copy : (len - start);
        size_t end = start + real_len;
        std::shared_ptr<char> build_values(
            new char[feature_value_size * real_len],
            [](char* p) { delete[] p; });
        uint64_t offset = start * feature_value_size;
        char* test_build_values = build_values.get();

        cudaMemcpyAsync(test_build_values,
                        hbm_pool->mem() + offset,
                        feature_value_size * real_len,
                        cudaMemcpyDeviceToHost,
                        stream);
        for (size_t k = 0; k < real_len; k = k + once_cpu_num) {
          struct task_info task;
          task.build_values = build_values;
          task.offset = start;
          task.device_id = i;
          task.multi_mf_dim = j;
          task.start = k;
          task.end =
              (k + once_cpu_num) < real_len ? (k + once_cpu_num) : (real_len);
          cpu_reday_channels_[i]->Put(task);
        }
        start = end;
      }
      total_len += len;
    }

    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    tm.Pause();
    VLOG(1) << "dump_pool_to_cpu_func i=" << i << ", total len=" << total_len
            << ", span=" << tm.ElapsedSec();
  };
  auto cpu_func = [this, &accessor_wrapper_ptr](int j) {
    struct task_info task;
    while (cpu_reday_channels_[j]->Get(task)) {
      auto& device_keys =
          this->current_task_
              ->device_dim_keys_[task.device_id][task.multi_mf_dim];
      uint64_t unuse_key = std::numeric_limits<uint64_t>::max();
      for (int i = task.start; i < task.end; ++i) {
        if (device_keys[i + task.offset] == unuse_key) {
          continue;
        }
#ifdef PADDLE_WITH_PSLIB
// TODO(lxsbupt): PSLIB DumpFill
#endif
#ifdef PADDLE_WITH_PSCORE
        int mf_dim = this->index_dim_vec_[task.multi_mf_dim];
        char* test_build_values = task.build_values.get();
        size_t feature_value_size =
            accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);
        size_t local_offset = i * feature_value_size;
        float* gpu_val =
            reinterpret_cast<float*>(test_build_values + local_offset);
        accessor_wrapper_ptr->DumpFill(gpu_val, cpu_table_accessor_, mf_dim);
#endif
      }
    }
  };
  platform::Timer timer;
  timer.Start();
  std::vector<std::future<void>> cpu_task_futures;
  std::vector<std::future<void>> gpu_task_futures;
  size_t device_num = heter_devices_.size();
  if (multi_mf_dim_) {
    VLOG(1) << "psgpu wrapper dump pool: multi_mf_dim_: " << multi_mf_dim_;
    for (size_t i = 0; i < device_num; i++) {
      cpu_reday_channels_[i]->Open();
      gpu_task_futures.emplace_back(hbm_thread_pool_[i]->enqueue(
          dump_pool_to_cpu_func, i, once_gpu_copy));
      for (size_t j = 0; j < cpu_device_thread_num_; j++) {
        cpu_task_futures.emplace_back(cpu_work_pool_[i]->enqueue(cpu_func, i));
      }
    }
  }
  for (auto& f : gpu_task_futures) {
    f.wait();
  }
  timer.Pause();
  VLOG(1) << "passid=" << current_task_->pass_id_
          << ", EndPass  dump_pool_to_cpu_func "
          << " cost " << timer.ElapsedSec() << " s.";
  for (size_t i = 0; i < device_num; i++) {
    cpu_reday_channels_[i]->Close();
  }
  gpu_task_futures.clear();
  timer.Start();
  for (auto& f : cpu_task_futures) {
    f.wait();
  }
  cpu_task_futures.clear();
  timer.Pause();
  VLOG(1) << "passid=" << current_task_->pass_id_ << ", EndPass  cpu_func "
          << " cost " << timer.ElapsedSec() << " s.";
  if (keysize_max != 0) {
    HeterPs_->end_pass();
  }
}

void PSGPUWrapper::DumpToMem() {
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
  if (gpu_graph_mode_ &&
      (FLAGS_gpugraph_storage_mode == GpuGraphStorageMode::WHOLE_HBM)) {
    this->HbmToSparseTable();
  }
#endif
}

void PSGPUWrapper::PullSparse(const phi::Place& place,
                              const int table_id,
                              const std::vector<const uint64_t*>& keys,
                              const std::vector<float*>& values,
                              const std::vector<int64_t>& slot_lengths,
                              const int hidden_size) {
  VLOG(0) << "Warning:: recommend use pull_gpups_sparse op instead. This "
             "PullSparse is not used.";
}

void PSGPUWrapper::PullSparse(const phi::Place& place,
                              const int table_id,
                              const std::vector<const uint64_t*>& keys,
                              const std::vector<float*>& values,
                              const std::vector<int64_t>& slot_lengths,
                              const std::vector<int>& slot_dim,
                              const int hidden_size) {
  VLOG(3) << "Begin Gpu Ps PullSparse";
  platform::Timer all_timer;
  platform::Timer pull_gpups_timer;
  all_timer.Start();

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t feature_value_size =
      accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);
  VLOG(3) << "PullSparse max_dim:" << max_mf_dim_
          << " pull_feature_value_size:" << pull_type_size_;

  if (phi::is_cpu_place(place)) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GpuPs now."));
  } else if (phi::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    int device_id = place.GetDeviceId();
    platform::CUDADeviceGuard guard(device_id);
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    if (FLAGS_gpugraph_dedup_pull_push_mode > 0) {
      auto& dev = device_caches_[devid_2_index];
      int slot_num = static_cast<int>(slot_lengths.size());
      std::vector<int64_t> slot_lengths_lod;
      slot_lengths_lod.reserve(slot_num + 1);
      slot_lengths_lod.push_back(0);

      int64_t total_length = 0;
      for (int i = 0; i < slot_num; ++i) {
        total_length += slot_lengths[i];
        slot_lengths_lod.push_back(total_length);
      }
      dev.total_key_length = total_length;
      VLOG(3) << "[" << device_id << "]Begin copy keys, key_num["
              << total_length << "] dedup mode";

      auto stream = dynamic_cast<phi::GPUContext*>(
                        phi::DeviceContextPool::Instance().Get(place))
                        ->stream();

      uint64_t* total_keys = dev.keys_tensor.mutable_data<uint64_t>(
          (total_length * 3) * sizeof(uint64_t), place);

      int* gpu_slot_dims = dev.dims_tensor.mutable_data<int>(
          slot_dim.size() * sizeof(int), place);
      uint64_t** gpu_keys = dev.keys_ptr_tensor.mutable_data<uint64_t*>(
          keys.size() * sizeof(uint64_t*), place);

      int64_t* slot_lens = dev.slot_lens.mutable_data<int64_t>(
          (slot_num + 1) * sizeof(int64_t), place);
      cudaMemcpyAsync(gpu_keys,
                      keys.data(),
                      keys.size() * sizeof(uint64_t*),
                      cudaMemcpyHostToDevice,
                      stream);
      cudaMemcpyAsync(slot_lens,
                      slot_lengths_lod.data(),
                      slot_lengths_lod.size() * sizeof(int64_t),
                      cudaMemcpyHostToDevice,
                      stream);

      cudaMemcpyAsync(gpu_slot_dims,
                      slot_dim.data(),
                      slot_dim.size() * sizeof(int),
                      cudaMemcpyHostToDevice,
                      stream);
      float** gpu_values = dev.values_ptr_tensor.mutable_data<float*>(
          values.size() * sizeof(float*), place);
      cudaMemcpyAsync(gpu_values,
                      values.data(),
                      values.size() * sizeof(float*),
                      cudaMemcpyHostToDevice,
                      stream);

      int* key2slot = dev.keys2slot.mutable_data<int>(
          (total_length * 5) * sizeof(int), place);

      this->CopyKeys(place,
                     gpu_keys,
                     total_keys,
                     slot_lens,
                     slot_num,
                     static_cast<int>(total_length),
                     key2slot);

      uint32_t* d_restore_idx =
          reinterpret_cast<uint32_t*>(&key2slot[total_length]);
      uint32_t* d_sorted_idx =
          reinterpret_cast<uint32_t*>(&d_restore_idx[total_length]);
      uint32_t* d_offset =
          reinterpret_cast<uint32_t*>(&d_sorted_idx[total_length]);
      uint32_t* d_merged_cnts =
          reinterpret_cast<uint32_t*>(&d_offset[total_length]);
      uint64_t* d_merged_keys =
          reinterpret_cast<uint64_t*>(&total_keys[total_length]);
      uint64_t* d_sorted_keys =
          reinterpret_cast<uint64_t*>(&d_merged_keys[total_length]);

      int dedup_size = HeterPs_->dedup_keys_and_fillidx(
          devid_2_index,
          static_cast<int>(total_length),
          total_keys,     // input
          d_merged_keys,  // output
          d_sorted_keys,  // sort keys
          d_restore_idx,  // pull fill idx
          d_sorted_idx,   // sort old idx
          d_offset,       // offset
          d_merged_cnts,
          FLAGS_gpugraph_dedup_pull_push_mode & 0x02);
      //      printf("device %d, end dedup_keys_and_fillidx total %d, "
      //              "dedup_size %d, slot num: %d, value size: %d\n",
      //             device_id, int(total_length), dedup_size, slot_num,
      //             int(feature_value_size));

      // uint64_t h_merged_keys[total_length];
      // cudaMemcpyAsync(h_merged_keys,
      //                 total_keys,
      //                 total_length * sizeof(uint64_t),
      //                 cudaMemcpyDeviceToHost,
      //                 stream);
      // cudaStreamSynchronize(stream);
      // std::stringstream ss;
      // for (int i = 0; i < total_length; ++i) {
      //   ss << h_merged_keys[i] << " ";
      // }
      // VLOG(1) << "h_merged_keys:" << ss.str();

      PADDLE_ENFORCE_GT(dedup_size,
                        0,
                        common::errors::PreconditionNotMet(
                            "dedup keys need more than zero failed in BoxPS."));
      dev.dedup_key_length = dedup_size;

      int64_t total_bytes = dedup_size * feature_value_size;
      float* total_values_gpu =
          dev.pull_push_tensor.mutable_data<float>(total_bytes, place);
      pull_gpups_timer.Start();
      HeterPs_->pull_sparse(
          devid_2_index, d_merged_keys, total_values_gpu, dedup_size);

      // values.size() not sure equal slot_num
      accessor_wrapper_ptr->CopyForPull(place,
                                        total_keys,
                                        gpu_values,
                                        total_values_gpu,
                                        slot_lens,
                                        key2slot,
                                        max_mf_dim_ + 3,
                                        total_length,
                                        gpu_slot_dims,
                                        d_restore_idx,
                                        feature_value_size);

      // float* h_values[dedup_size];
      // cudaMemcpyAsync(&h_values,
      //                 total_values_gpu,
      //                 total_bytes,
      //                 cudaMemcpyDeviceToHost,
      //                 stream);
      // cudaStreamSynchronize(stream);
      // std::stringstream ss;
      // std::stringstream ss1;
      // for (int i = 0; i < dedup_size; ++i) {
      //   ss << i << ": ";
      //   for (size_t j = 0; j < feature_value_size/ sizeof(float); ++j) {
      //     auto v =  *(reinterpret_cast<float*>(h_values[i] + j));
      //     ss <<  v << " ";
      //     if (v > 100000 || v < -100000) {
      //       ss1 << i << "-" << j << ":" << v << ";";
      //     }
      //   }
      //   ss <<";";
      // }
      // VLOG(1) << "h_values:" << ss.str();
      // VLOG(1) << "not normal h_values:" << ss1.str();
    } else {
      size_t total_length =
          std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
      auto buf = memory::Alloc(place, total_length * feature_value_size);
      float* total_values_gpu = reinterpret_cast<float*>(buf->ptr());
      VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
      phi::DenseTensor& total_keys_tensor = keys_tensor[devid_2_index];
      uint64_t* total_keys =
          reinterpret_cast<uint64_t*>(total_keys_tensor.mutable_data<int64_t>(
              {int64_t(total_length), 1}, place));
      // construct slot_level lod info
      auto slot_lengths_lod = slot_lengths;
      for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
        slot_lengths_lod[i] += slot_lengths_lod[i - 1];
      }
      auto buf_key = memory::Alloc(place, keys.size() * sizeof(uint64_t*));
      auto buf_length =
          memory::Alloc(place, slot_lengths.size() * sizeof(int64_t));
      uint64_t** gpu_keys = reinterpret_cast<uint64_t**>(buf_key->ptr());
      int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
      cudaMemcpy(gpu_keys,
                 keys.data(),
                 keys.size() * sizeof(uint64_t*),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_len,
                 slot_lengths_lod.data(),
                 slot_lengths.size() * sizeof(int64_t),
                 cudaMemcpyHostToDevice);

      auto buf_dim = memory::Alloc(place, slot_dim.size() * sizeof(int));
      int* gpu_dim = reinterpret_cast<int*>(buf_dim->ptr());
      cudaMemcpy(gpu_dim,
                 slot_dim.data(),
                 slot_dim.size() * sizeof(int),
                 cudaMemcpyHostToDevice);

      this->CopyKeys(place,
                     gpu_keys,
                     total_keys,
                     gpu_len,
                     static_cast<int>(slot_lengths.size()),
                     static_cast<int>(total_length));
      VLOG(3) << "Begin call PullSparseGPU in GPUPS, dev: " << devid_2_index
              << " len: " << total_length;

      pull_gpups_timer.Start();
      HeterPs_->pull_sparse(
          devid_2_index, total_keys, total_values_gpu, total_length);

      VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
              << "]";

      accessor_wrapper_ptr->CopyForPull(place,
                                        gpu_keys,
                                        values,
                                        total_values_gpu,
                                        gpu_len,
                                        static_cast<int>(slot_lengths.size()),
                                        hidden_size,
                                        total_length,
                                        gpu_dim,
                                        feature_value_size);
    }
    pull_gpups_timer.Pause();
#endif
  } else if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU_KP
    VLOG(3) << "Begin Xpu Ps PullSparse";
    size_t total_length =
        std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
    FeatureValue* total_values_gpu = nullptr;
    xpu_malloc(reinterpret_cast<void**>(&total_values_gpu),
               total_length * feature_value_size);
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    phi::DenseTensor& total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys =
        reinterpret_cast<uint64_t*>(total_keys_tensor.mutable_data<int64_t>(
            {int64_t(total_length), 1}, place));

    // construct slot_level lod info
    auto slot_lengths_lod = slot_lengths;
    for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
      slot_lengths_lod[i] += slot_lengths_lod[i - 1];
    }

    auto buf_key = memory::Alloc(place, keys.size() * sizeof(uint64_t*));
    auto buf_length =
        memory::Alloc(place, slot_lengths.size() * sizeof(int64_t));
    uint64_t** xpu_keys = reinterpret_cast<uint64_t**>(buf_key->ptr());
    int64_t* xpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(xpu_keys,
                                          keys.data(),
                                          keys.size() * sizeof(uint64_t*),
                                          XPU_HOST_TO_DEVICE));
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(xpu_len,
                                          slot_lengths_lod.data(),
                                          slot_lengths.size() * sizeof(int64_t),
                                          XPU_HOST_TO_DEVICE));

    this->CopyKeys(place,
                   xpu_keys,
                   total_keys,
                   xpu_len,
                   static_cast<int>(slot_lengths.size()),
                   static_cast<int>(total_length));
    VLOG(3) << "Begin call PullSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    pull_gpups_timer.Start();
    HeterPs_->pull_sparse(devid_2_index,
                          total_keys,
                          total_values_gpu,
                          static_cast<int>(total_length));
    pull_gpups_timer.Pause();

    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";
    accessor_wrapper_ptr->CopyForPull(place,
                                      xpu_keys,
                                      values,
                                      total_values_gpu,
                                      xpu_len,
                                      static_cast<int>(slot_lengths.size()),
                                      hidden_size,
                                      total_length,
                                      feature_value_size);
#endif
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "GpuPs/XpuPs: PullSparse Only Support CUDAPlace or XPUPlace Now."));
  }
  all_timer.Pause();
  VLOG(3) << "GpuPs PullSparse total costs: " << all_timer.ElapsedSec()
          << " s, of which GPUPS costs: " << pull_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PullSparse";
}

void PSGPUWrapper::PushSparseGrad(const phi::Place& place,
                                  const int table_id,
                                  const std::vector<const uint64_t*>& keys,
                                  const std::vector<const float*>& grad_values,
                                  const std::vector<int64_t>& slot_lengths,
                                  const int hidden_size,
                                  const int batch_size) {
  ++grad_push_count_;
  platform::Timer all_timer;
  platform::Timer push_gpups_timer;
  all_timer.Start();
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  if (phi::is_cpu_place(place)) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GPUPS now."));
  } else if (phi::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    int device_id = place.GetDeviceId();
    platform::CUDADeviceGuard guard(device_id);
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    if (FLAGS_gpugraph_dedup_pull_push_mode > 0) {
      auto& dev = device_caches_[devid_2_index];
      int64_t total_length = dev.total_key_length;
      VLOG(3) << "Begin push sparse, key_num[" << total_length
              << "] dedup mode, device:" << device_id << ", index"
              << devid_2_index;
      auto stream = dynamic_cast<phi::GPUContext*>(
                        phi::DeviceContextPool::Instance().Get(place))
                        ->stream();
      uint64_t* total_keys = dev.keys_tensor.data<uint64_t>();
      int* slot_dims = dev.dims_tensor.data<int>();
      int slot_num = static_cast<int>(slot_lengths.size());
      if (!dev.d_slot_vector.IsInitialized()) {
        int* buf_slot_vector =
            dev.d_slot_vector.mutable_data<int>(slot_num * sizeof(int), place);
        cudaMemcpyAsync(buf_slot_vector,
                        slot_vector_.data(),
                        slot_num * sizeof(int),
                        cudaMemcpyHostToDevice,
                        stream);
      }

      const int64_t* slot_lens = dev.slot_lens.data<int64_t>();
      const int* d_slot_vector = dev.d_slot_vector.data<int>();
      const int* key2slot = dev.keys2slot.data<int>();
      float** gpu_values = dev.values_ptr_tensor.data<float*>();
      cudaMemcpyAsync(gpu_values,
                      grad_values.data(),
                      grad_values.size() * sizeof(float*),
                      cudaMemcpyHostToDevice,
                      stream);

      uint64_t* d_merged_keys = &total_keys[total_length];

      int64_t dedup_size = dev.dedup_key_length;
      int64_t total_bytes = dedup_size * grad_value_size;
      float* total_grad_values_gpu =
          dev.pull_push_tensor.mutable_data<float>(total_bytes, place);
      // dedup rate more than 3
      if (total_length > dedup_size * 3) {
        const uint32_t* d_restore_idx =
            reinterpret_cast<const uint32_t*>(&key2slot[total_length]);
        accessor_wrapper_ptr->CopyForPush(place,
                                          total_keys,
                                          gpu_values,
                                          total_grad_values_gpu,
                                          d_slot_vector,
                                          slot_lens,
                                          max_mf_dim_ + 3,
                                          total_length,
                                          dedup_size,
                                          batch_size,
                                          slot_dims,
                                          key2slot,
                                          d_restore_idx,
                                          grad_value_size);
      } else {
        const uint32_t* d_sorted_idx =
            reinterpret_cast<const uint32_t*>(&key2slot[total_length * 2]);
        const uint32_t* d_offset =
            reinterpret_cast<const uint32_t*>(&d_sorted_idx[total_length]);
        const uint32_t* d_merged_cnts =
            reinterpret_cast<const uint32_t*>(&d_offset[total_length]);
        accessor_wrapper_ptr->CopyForPush(place,
                                          d_merged_keys,
                                          gpu_values,
                                          total_grad_values_gpu,
                                          d_slot_vector,
                                          slot_lens,
                                          max_mf_dim_ + 3,
                                          total_length,
                                          dedup_size,
                                          batch_size,
                                          slot_dims,
                                          key2slot,
                                          d_sorted_idx,
                                          d_offset,
                                          d_merged_cnts,
                                          grad_value_size);
      }

      push_gpups_timer.Start();
      HeterPs_->push_sparse(devid_2_index,
                            d_merged_keys,
                            total_grad_values_gpu,
                            static_cast<int>(dedup_size));
    } else {
      int64_t total_length =
          std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
      VLOG(3) << "Begin GPUPS PushSparseGrad";

      auto buf = memory::Alloc(place, total_length * grad_value_size);
      VLOG(3) << "Push Sparse Max mf dimension: " << max_mf_dim_
              << "grad_value_size:" << grad_value_size;
      float* total_grad_values_gpu = reinterpret_cast<float*>(buf->ptr());

      phi::DenseTensor& total_keys_tensor = keys_tensor[devid_2_index];
      uint64_t* total_keys =
          reinterpret_cast<uint64_t*>(total_keys_tensor.data<int64_t>());
      VLOG(3) << "Begin copy grad tensor to gpups struct";

      accessor_wrapper_ptr->CopyForPush(place,
                                        grad_values,
                                        total_grad_values_gpu,
                                        slot_lengths,
                                        total_length,
                                        batch_size,
                                        grad_value_size,
                                        slot_vector_,
                                        slot_mf_dim_vector_);

      VLOG(3) << "Begin call PushSparseGPU in GPUPS, dev: " << devid_2_index
              << " len: " << total_length;
      push_gpups_timer.Start();
      HeterPs_->push_sparse(devid_2_index,
                            total_keys,
                            total_grad_values_gpu,
                            static_cast<int>(total_length));
    }
    push_gpups_timer.Pause();
#endif
  } else if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU_KP
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    int64_t total_length =
        std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
    VLOG(3) << "Begin GPUPS PushSparseGrad";

    auto buf = memory::Alloc(place, total_length * grad_value_size);
    VLOG(3) << "Push Sparse Max mf dimension: " << max_mf_dim_
            << "grad_value_size:" << grad_value_size;
    float* total_grad_values_gpu = reinterpret_cast<float*>(buf->ptr());
    phi::DenseTensor& total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys =
        reinterpret_cast<uint64_t*>(total_keys_tensor.data<int64_t>());
    VLOG(3) << "Begin copy grad tensor to xpups struct";
    accessor_wrapper_ptr->CopyForPush(place,
                                      grad_values,
                                      total_grad_values_gpu,
                                      slot_lengths,
                                      hidden_size,
                                      total_length,
                                      batch_size,
                                      slot_vector_);

    VLOG(3) << "Begin call PushSparseXPU in XPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    push_gpups_timer.Start();
    HeterPs_->push_sparse(devid_2_index,
                          total_keys,
                          total_grad_values_gpu,
                          static_cast<int>(total_length));
    push_gpups_timer.Pause();
#endif
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "GPUPS: PushSparseGrad Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  time_3 += all_timer.ElapsedSec();
  time_4 += push_gpups_timer.ElapsedSec();
  VLOG(3) << "PushSparseGrad total cost: " << all_timer.ElapsedSec()
          << " s, of which GPUPS cost: " << push_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PushSparseGrad";
}

}  // namespace paddle::framework
#endif
