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

#include <algorithm>
#include <deque>

#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/platform/timer.h"

#include "paddle/fluid/framework/fleet/heter_ps/heter_ps.h"

namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_PSLIB
void AfsWrapper::init(const std::string& fs_name, const std::string& fs_user,
                      const std::string& pass_wd, const std::string& conf) {
  int ret = afs_handler_.init(fs_name.c_str(), fs_user.c_str(), pass_wd.c_str(),
                              conf.c_str());
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
#ifdef PADDLE_WITH_PSLIB
void PSGPUWrapper::InitAfsApi(const std::string& fs_name,
                              const std::string& fs_user,
                              const std::string& pass_wd,
                              const std::string& conf) {
  int ret = afs_handler_.init(fs_name.c_str(), fs_user.c_str(), pass_wd.c_str(),
                              conf.c_str());
  if (ret != 0) {
    VLOG(0) << "AFS Init Error";
  }
  use_afs_api_ = 1;
}
#endif
void PSGPUWrapper::PreBuildTask(std::shared_ptr<HeterContext> gpu_task) {
  VLOG(3) << "PSGPUWrapper::BuildGPUPSTask begin";
  platform::Timer timeline;
  timeline.Start();
  int device_num = heter_devices_.size();
  if (!multi_mf_dim_) {
    gpu_task->init(thread_keys_shard_num_, device_num);
  } else {
    gpu_task->init(thread_keys_shard_num_, device_num, multi_mf_dim_);
  }
  auto& local_keys = gpu_task->feature_keys_;
  auto& local_ptr = gpu_task->value_ptr_;

  std::vector<std::thread> threads;

  // data should be in input channel
  if (!multi_mf_dim_) {
    thread_keys_.resize(thread_keys_thread_num_);
    for (int i = 0; i < thread_keys_thread_num_; i++) {
      thread_keys_[i].resize(thread_keys_shard_num_);
    }
  } else {
    thread_dim_keys_.resize(thread_keys_thread_num_);
    for (int i = 0; i < thread_keys_thread_num_; i++) {
      thread_dim_keys_[i].resize(thread_keys_shard_num_);
      for (int j = 0; j < thread_keys_shard_num_; j++) {
        thread_dim_keys_[i][j].resize(multi_mf_dim_);
      }
    }
  }

  size_t total_len = 0;
  size_t len_per_thread = 0;
  int remain = 0;
  size_t begin = 0;

  std::string data_set_name = std::string(typeid(*dataset_).name());

  if (data_set_name.find("SlotRecordDataset") != std::string::npos) {
    VLOG(0) << "ps_gpu_wrapper use SlotRecordDataset";
    SlotRecordDataset* dataset = dynamic_cast<SlotRecordDataset*>(dataset_);
    auto input_channel = dataset->GetInputChannel();
    VLOG(0) << "yxf::buildtask::inputslotchannle size: "
            << input_channel->Size();
    const std::deque<SlotRecord>& vec_data = input_channel->GetData();
    total_len = vec_data.size();
    len_per_thread = total_len / thread_keys_thread_num_;
    remain = total_len % thread_keys_thread_num_;
    VLOG(0) << "total len: " << total_len;
    auto gen_func = [this](const std::deque<SlotRecord>& total_data,
                           int begin_index, int end_index, int i) {
      for (auto iter = total_data.begin() + begin_index;
           iter != total_data.begin() + end_index; iter++) {
        const auto& ins = *iter;
        const auto& feasign_v = ins->slot_uint64_feasigns_.slot_values;
        for (const auto feasign : feasign_v) {
          int shard_id = feasign % thread_keys_shard_num_;
          this->thread_keys_[i][shard_id].insert(feasign);
        }
      }
    };
    auto gen_dynamic_mf_func = [this](const std::deque<SlotRecord>& total_data,
                                      int begin_index, int end_index, int i) {
      for (auto iter = total_data.begin() + begin_index;
           iter != total_data.begin() + end_index; iter++) {
        const auto& ins = *iter;
        const auto& feasign_v = ins->slot_uint64_feasigns_.slot_values;
        const auto& slot_offset = ins->slot_uint64_feasigns_.slot_offsets;
        for (size_t slot_idx = 0; slot_idx < slot_offset_vector_.size();
             slot_idx++) {
          for (size_t j = slot_offset[slot_offset_vector_[slot_idx]];
               j < slot_offset[slot_offset_vector_[slot_idx] + 1]; j++) {
            int shard_id = feasign_v[j] % thread_keys_shard_num_;
            int dim_id = slot_index_vec_[slot_idx];
            this->thread_dim_keys_[i][shard_id][dim_id].insert(feasign_v[j]);
          }
        }
      }
      /*
      for (auto iter = total_data.begin() + begin_index;
           iter != total_data.begin() + end_index; iter++) {
        const auto& ins = *iter;
        const auto& feasign_v = ins->slot_uint64_feasigns_.slot_values;
        for (const auto feasign : feasign_v) {
          int shard_id = feasign % thread_keys_shard_num_;
          this->thread_dim_keys_[i][shard_id][0].insert(feasign);
        }
      }
      */
    };
    for (int i = 0; i < thread_keys_thread_num_; i++) {
      if (!multi_mf_dim_) {
        VLOG(0) << "yxf::psgpu wrapper genfunc";
        threads.push_back(
            std::thread(gen_func, std::ref(vec_data), begin,
                        begin + len_per_thread + (i < remain ? 1 : 0), i));
      } else {
        VLOG(0) << "yxf::psgpu wrapper genfunc with dynamic mf";
        threads.push_back(
            std::thread(gen_dynamic_mf_func, std::ref(vec_data), begin,
                        begin + len_per_thread + (i < remain ? 1 : 0), i));
      }
      begin += len_per_thread + (i < remain ? 1 : 0);
    }
    for (std::thread& t : threads) {
      t.join();
    }
    timeline.Pause();
    VLOG(0) << "GpuPs build task cost " << timeline.ElapsedSec() << " seconds.";
  } else {
    CHECK(data_set_name.find("MultiSlotDataset") != std::string::npos);
    VLOG(0) << "ps_gpu_wrapper use MultiSlotDataset";
    MultiSlotDataset* dataset = dynamic_cast<MultiSlotDataset*>(dataset_);
    auto input_channel = dataset->GetInputChannel();

    const std::deque<Record>& vec_data = input_channel->GetData();
    total_len = vec_data.size();
    len_per_thread = total_len / thread_keys_thread_num_;
    remain = total_len % thread_keys_thread_num_;
    auto gen_func = [this](const std::deque<Record>& total_data,
                           int begin_index, int end_index, int i) {
      for (auto iter = total_data.begin() + begin_index;
           iter != total_data.begin() + end_index; iter++) {
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
          std::thread(gen_func, std::ref(vec_data), begin,
                      begin + len_per_thread + (i < remain ? 1 : 0), i));
      begin += len_per_thread + (i < remain ? 1 : 0);
    }
    for (std::thread& t : threads) {
      t.join();
    }
    timeline.Pause();
    VLOG(0) << "GpuPs build task cost " << timeline.ElapsedSec() << " seconds.";
  }

  timeline.Start();

  threads.clear();
  // merge thread_keys to shard_keys
  auto merge_ins_func = [this, gpu_task](int shard_num) {
    for (int i = 0; i < thread_keys_thread_num_; ++i) {
      gpu_task->batch_add_keys(shard_num, thread_keys_[i][shard_num]);
      thread_keys_[i][shard_num].clear();
    }
  };
  auto merge_ins_dynamic_mf_func = [this, gpu_task](int shard_num, int dim_id) {
    for (int i = 0; i < thread_keys_thread_num_; ++i) {
      gpu_task->batch_add_keys(shard_num, dim_id,
                               thread_dim_keys_[i][shard_num][dim_id]);
      thread_dim_keys_[i][shard_num][dim_id].clear();
    }
  };
  // for (size_t i = 0; i < thread_keys_.size(); i++) {
  //  gpu_task->batch_add_keys(thread_keys_[i]);
  //  for (int j = 0; j < thread_keys_thread_num_; j++) {
  //    thread_keys_[i][j].clear();
  //  }
  //}
  for (int i = 0; i < thread_keys_shard_num_; ++i) {
    if (!multi_mf_dim_) {
      threads.push_back(std::thread(merge_ins_func, i));
    } else {
      for (int j = 0; j < multi_mf_dim_; j++) {
        threads.push_back(std::thread(merge_ins_dynamic_mf_func, i, j));
      }
    }
  }
  for (auto& t : threads) {
    t.join();
  }
  timeline.Pause();

  VLOG(0) << "GpuPs task add keys cost " << timeline.ElapsedSec()
          << " seconds.";
  timeline.Start();
  gpu_task->UniqueKeys();
  timeline.Pause();

  VLOG(0) << "GpuPs task unique cost " << timeline.ElapsedSec() << " seconds.";

  if (!multi_mf_dim_) {
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      VLOG(0) << "GpuPs shard: " << i << " key len: " << local_keys[i].size();
      local_ptr[i].resize(local_keys[i].size());
    }
  } else {
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        VLOG(0) << "GpuPs shard: " << i << "mf dim: " << index_dim_vec_[j]
                << " key len: " << gpu_task->feature_dim_keys_[i][j].size();
        gpu_task->value_dim_ptr_[i][j].resize(
            gpu_task->feature_dim_keys_[i][j].size());
      }
    }
  }
}

void PSGPUWrapper::BuildPull(std::shared_ptr<HeterContext> gpu_task) {
  platform::Timer timeline;
  std::vector<std::future<void>> task_futures;
  int device_num = heter_devices_.size();
  auto& local_keys = gpu_task->feature_keys_;
  auto& local_ptr = gpu_task->value_ptr_;

  auto& local_dim_keys = gpu_task->feature_dim_keys_;
  auto& local_dim_ptr = gpu_task->value_dim_ptr_;

  auto& device_keys = gpu_task->device_keys_;
  auto& device_vals = gpu_task->device_values_;
  auto& device_dim_keys = gpu_task->device_dim_keys_;
  auto& device_dim_ptr = gpu_task->device_dim_ptr_;
  auto& device_dim_mutex = gpu_task->dim_mutex_;
  if (multi_mf_dim_) {
    for (size_t dev = 0; dev < device_dim_keys.size(); dev++) {
      device_dim_keys[dev].resize(multi_mf_dim_);
      device_dim_ptr[dev].resize(multi_mf_dim_);
    }
  }
  // auto& device_mutex = gpu_task->mutex_;

  std::vector<std::thread> threads(thread_keys_shard_num_);
#ifdef PADDLE_WITH_PSLIB
  auto fleet_ptr = FleetWrapper::GetInstance();
#endif
#ifdef PADDLE_WITH_PSCORE
  auto fleet_ptr = paddle::distributed::FleetWrapper::GetInstance();
#endif

#if (defined PADDLE_WITH_PSLIB) && (defined PADDLE_WITH_HETERPS)
  // get day_id: day nums from 1970
  struct std::tm b;
  b.tm_year = year_ - 1900;
  b.tm_mon = month_ - 1;
  b.tm_mday = day_;
  b.tm_min = b.tm_hour = b.tm_sec = 0;
  std::time_t seconds_from_1970 = std::mktime(&b);
  int day_id = seconds_from_1970 / 86400;
  fleet_ptr->pslib_ptr_->_worker_ptr->set_day_id(table_id_, day_id);
#endif

  timeline.Start();
  auto ptl_func = [this, &local_keys, &local_ptr, &fleet_ptr](int i) {
    size_t key_size = local_keys[i].size();
    int32_t status = -1;
#ifdef PADDLE_WITH_PSLIB
    // auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(
    //    reinterpret_cast<char**>(local_ptr[i].data()), this->table_id_,
    //    local_keys[i].data(), key_size);
    int32_t cnt = 0;
    while (true) {
      auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(
          i, reinterpret_cast<char**>(local_ptr[i].data()), this->table_id_,
          local_keys[i].data(), key_size);
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
    int32_t cnt = 0;
    while (true) {
      auto tt = fleet_ptr->worker_ptr_->PullSparsePtr(
          reinterpret_cast<char**>(local_ptr[i].data()), this->table_id_,
          local_keys[i].data(), key_size);
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
      VLOG(3) << "FleetWrapper Pull sparse to local done with table size: "
              << local_keys[i].size();
    }
  };

  auto ptl_dynamic_mf_func = [this, &local_dim_keys, &local_dim_ptr,
                              &fleet_ptr](int i, int j) {
#ifdef PADDLE_WITH_PSLIB
    size_t key_size = local_dim_keys[i][j].size();
    int32_t status = -1;
    int32_t cnt = 0;
    while (true) {
      auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(
          i, reinterpret_cast<char**>(local_dim_ptr[i][j].data()),
          this->table_id_, local_dim_keys[i][j].data(), key_size);
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
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(300);
      exit(-1);
    } else {
      VLOG(0) << "FleetWrapper Pull sparse to local done with table size: "
              << local_dim_keys[i][j].size();
    }
#endif
  };
  if (!multi_mf_dim_) {
    for (size_t i = 0; i < threads.size(); i++) {
      threads[i] = std::thread(ptl_func, i);
    }
  } else {
    threads.resize(thread_keys_shard_num_ * multi_mf_dim_);
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        threads[i * multi_mf_dim_ + j] = std::thread(ptl_dynamic_mf_func, i, j);
      }
    }
  }
  for (std::thread& t : threads) {
    t.join();
  }
  timeline.Pause();
  VLOG(0) << "pull sparse from CpuPS into GpuPS cost " << timeline.ElapsedSec()
          << " seconds.";
  if (multi_node_) {
    auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
    if (!gloo_wrapper->IsInitialized()) {
      VLOG(0) << "GLOO is not inited";
      gloo_wrapper->Init();
    }
    gloo_wrapper->Barrier();
  }

  timeline.Start();
  std::vector<std::vector<std::pair<uint64_t, char*>>> pass_values;

  bool record_status = false;
#ifdef PADDLE_WITH_PSLIB
  uint16_t pass_id = 0;
  if (multi_node_) {
    record_status = fleet_ptr->pslib_ptr_->_worker_ptr->take_sparse_record(
        table_id_, pass_id, pass_values);
  }
#endif
  auto& device_task_keys = gpu_task->device_task_keys_;
  auto& device_task_ptrs = gpu_task->device_task_ptr_;
  auto build_dynamic_mf_func = [this, device_num, &local_dim_keys,
                                &local_dim_ptr, &device_dim_keys,
                                &device_dim_ptr,
                                &device_dim_mutex](int i, int j) {
#ifdef PADDLE_WITH_PSLIB
    std::vector<std::vector<FeatureKey>> task_keys(device_num);
    std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>> task_ptrs(
        device_num);
    for (size_t k = 0; k < local_dim_keys[i][j].size(); k++) {
      int shard = local_dim_keys[i][j][k] % device_num;
      task_keys[shard].push_back(local_dim_keys[i][j][k]);
      task_ptrs[shard].push_back(local_dim_ptr[i][j][k]);
    }
    for (int dev = 0; dev < device_num; dev++) {
      for (int dim = 0; dim < multi_mf_dim_; dim++) {
        device_dim_mutex[dev][dim]->lock();

        int len = task_keys[dev].size();
        int cur = device_dim_keys[dev][dim].size();
        device_dim_keys[dev][dim].resize(device_dim_keys[dev][dim].size() +
                                         len);
        device_dim_ptr[dev][dim].resize(device_dim_ptr[dev][dim].size() + len);
        for (int k = 0; k < len; ++k) {
          device_dim_keys[dev][dim][cur + k] = task_keys[dev][k];
          device_dim_ptr[dev][dim][cur + k] = task_ptrs[dev][k];
        }
        device_dim_mutex[dev][dim]->unlock();
      }
    }
#endif
  };
  auto build_func = [device_num, record_status, &pass_values, &local_keys,
                     &local_ptr, &device_task_keys, &device_task_ptrs](int i) {
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
      task_futures.emplace_back(hbm_thread_pool_[i]->enqueue(build_func, i));
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
  auto calc_prefix_func = [this, &prefix_sum, &device_keys, &device_vals,
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
  auto prepare_dev_value_func = [device_num, &prefix_sum, &device_keys,
                                 &device_vals, &device_task_keys,
                                 &device_task_ptrs](int dev, int shard_id) {
    auto& task_keys = device_task_keys[shard_id];
#ifdef PADDLE_WITH_PSLIB
    auto& task_ptrs = device_task_ptrs[shard_id];
#endif

#ifdef PADDLE_WITH_PSCORE
    auto& task_ptrs = device_task_ptrs[shard_id];
#endif

    int len = prefix_sum[dev][shard_id + 1] - prefix_sum[dev][shard_id];
    int cur = prefix_sum[dev][shard_id];
#ifdef PADDLE_WITH_PSLIB
    for (int j = 0; j < len; ++j) {
      device_keys[dev][cur + j] = task_keys[dev][j];
      float* ptr_val = task_ptrs[dev][j]->data();
      FeatureValue& val = device_vals[dev][cur + j];
      size_t dim = task_ptrs[dev][j]->size();

      val.delta_score = ptr_val[1];
      val.show = ptr_val[2];
      val.clk = ptr_val[3];
      val.slot = ptr_val[6];
      val.lr = ptr_val[4];
      val.lr_g2sum = ptr_val[5];
      val.cpu_ptr = (uint64_t)(task_ptrs[dev][j]);

      if (dim > 7) {
        val.mf_size = MF_DIM + 1;
        for (int x = 0; x < val.mf_size; x++) {
          val.mf[x] = ptr_val[x + 7];
        }
      } else {
        val.mf_size = 0;
        for (int x = 0; x < MF_DIM + 1; x++) {
          val.mf[x] = 0;
        }
      }
    }
#endif
#ifdef PADDLE_WITH_PSCORE
    for (int j = 0; j < len; ++j) {
      device_keys[dev][cur + j] = task_keys[dev][j];
      float* ptr_val = task_ptrs[dev][j]->data();
      FeatureValue& val = device_vals[dev][cur + j];
      size_t dim = task_ptrs[dev][j]->size();
      val.delta_score = ptr_val[2];
      val.show = ptr_val[3];
      val.clk = ptr_val[4];
      val.slot = ptr_val[0];
      val.lr = ptr_val[5];
      val.lr_g2sum = ptr_val[6];
      val.cpu_ptr = (uint64_t)(task_ptrs[dev][j]);

      if (dim > 7) {
        val.mf_size = MF_DIM + 1;
        for (int x = 0; x < val.mf_size; x++) {
          val.mf[x] = ptr_val[x + 7];
        }
      } else {
        val.mf_size = 0;
        for (int x = 0; x < MF_DIM + 1; x++) {
          val.mf[x] = 0;
        }
      }
    }
#endif
    VLOG(3) << "GpuPs build hbmps done";
  };

  if (multi_mf_dim_) {
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        threads[i * multi_mf_dim_ + j] =
            std::thread(build_dynamic_mf_func, i, j);
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
  } else {
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < device_num; j++) {
        task_futures.emplace_back(
            hbm_thread_pool_[i]->enqueue(prepare_dev_value_func, j, i));
      }
    }
    for (auto& f : task_futures) {
      f.wait();
    }
    task_futures.clear();
  }
  timeline.Pause();
  VLOG(0) << "GpuPs prepare for build hbm cost " << timeline.ElapsedSec()
          << " seconds.";
}

void PSGPUWrapper::BuildGPUTask(std::shared_ptr<HeterContext> gpu_task) {
  int device_num = heter_devices_.size();
  platform::Timer timeline;
  timeline.Start();

  std::vector<size_t> feature_keys_count(device_num);
  size_t size_max = 0;
  if (!multi_mf_dim_) {
    for (int i = 0; i < device_num; i++) {
      feature_keys_count[i] = gpu_task->device_keys_[i].size();
      VLOG(0) << i << " card contains feasign nums: " << feature_keys_count[i];
      size_max = std::max(size_max, feature_keys_count[i]);
    }
  } else {
    for (int i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        feature_keys_count[i] += gpu_task->device_dim_ptr_[i][j].size();
      }
      VLOG(0) << i << " card with dynamic mf contains feasign nums: "
              << feature_keys_count[i];
      size_max = std::max(size_max, feature_keys_count[i]);
    }
  }
  if (HeterPs_) {
    delete HeterPs_;
    HeterPs_ = nullptr;
  }
  if (size_max <= 0) {
    VLOG(0) << "Skip build gpu ps cause feasign nums = " << size_max;
    return;
  }
  std::vector<std::thread> threads(device_num);
  HeterPs_ = HeterPsBase::get_instance(size_max, resource_);
#ifdef PADDLE_WITH_CUDA
  HeterPs_->set_nccl_comm_and_size(inner_comms_, inter_comms_, node_size_);
#endif

#ifdef PADDLE_WITH_XPU_KP
  // build cachemanager
  // dangerous down cast the type of HeterPs_ is HeterPsBase*
  auto cache_manager = dynamic_cast<HeterPs*>(HeterPs_)->get_cache_manager();
  auto data_readers = dataset_->GetReaders(); 
  CHECK(data_readers.size() > 0);
  int batch_size = data_readers[0]->GetDefaultBatchSize();
  VLOG(0) << "BuildGPUTask: batchsize:" << batch_size
          << " device_num:" << device_num;
  cache_manager->init(12, batch_size, device_num);

  VLOG(0) << "BuildGPUTask: build_sign2fids";
  for (int i = 0; i < device_num; i++) {
    cache_manager->build_sign2fids(gpu_task->device_keys_[i].data(), gpu_task->device_keys_[i].size());
  }
  VLOG(0) << "BuildGPUTask: get inputchannel from dataset";
  // convert feasign to fid in dataset
  //SlotRecordDataset* dataset = dynamic_cast<SlotRecordDataset*>(dataset_);
  MultiSlotDataset* dataset = dynamic_cast<MultiSlotDataset*>(dataset_);
  CHECK(dataset != nullptr) << "only support MultiSlotDataset";
  auto input_channel = dataset->GetInputChannel();
  VLOG(0) << "BuildGPUTask: query sign2fid for dataset";

  // dangerous, the return value of GetData is unmutable in general
  //std::deque<SlotRecord>& vec_data = const_cast<std::deque<SlotRecord>&>(input_channel->GetData());
  std::deque<Record>& vec_data = const_cast<std::deque<Record>&>(input_channel->GetData());
  for (auto it = vec_data.begin(); it != vec_data.end(); it++) {
    //auto& feasign = ((*it)->slot_uint64_feasigns_).slot_values;
    //for (unsigned int j = 0; j < feasign.size(); j++) {
    //  feasign[j] = cache_manager->query_sign2fid(feasign[j]);
    //}
    auto & feasign = it->uint64_feasigns_;
    for (unsigned int j = 0; j < feasign.size(); j++) {
      feasign[j].sign().uint64_feasign_ = cache_manager -> query_sign2fid(feasign[j].sign().uint64_feasign_);
    }
  }
  VLOG(0) << "BuildGPUTask: query sign2fid for device keys";
  // convert feasign to fid in device_keys_
  for (int i = 0; i < device_num; i++) {
    auto& feasign = gpu_task->device_keys_[i];
    for (unsigned int j = 0; j < feasign.size(); j++) {
      feasign[j] = cache_manager->query_sign2fid(feasign[j]);
    }
  }
  VLOG(0) << "BuildGPUTask: exit PADDLE_WITH_XPU_KP";
#endif //PADDLE_WITH_XPU_KP
  auto build_func = [this, &gpu_task, &feature_keys_count](int i) {
    VLOG(3) << "building table: " << i;
    this->HeterPs_->build_ps(i, gpu_task->device_keys_[i].data(),
                             gpu_task->device_values_[i].data(),
                             feature_keys_count[i], 500000, 2);
    // if (feature_keys_count[i] > 0) {
    //   HeterPs_->show_one_table(i);
    // }
  };
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(build_func, i);
  }
  for (std::thread& t : threads) {
    t.join();
  }
  timeline.Pause();
  VLOG(0) << "GpuPs build table total costs: " << timeline.ElapsedSec()
          << " s.";
}

void PSGPUWrapper::LoadIntoMemory(bool is_shuffle) {
  platform::Timer timer;
  VLOG(3) << "Begin LoadIntoMemory(), dataset[" << dataset_ << "]";
  timer.Start();
  dataset_->LoadIntoMemory();
  timer.Pause();
  VLOG(0) << "LoadIntoMemory cost: " << timer.ElapsedSec() << "s";

  // local shuffle
  if (is_shuffle) {
    dataset_->LocalShuffle();
  }

  std::shared_ptr<HeterContext> gpu_task = gpu_task_pool_.Get();
  gpu_task->Reset();
  data_ready_channel_->Put(gpu_task);
  VLOG(3) << "End LoadIntoMemory(), dataset[" << dataset_ << "]";
}

void PSGPUWrapper::start_build_thread() {
  running_ = true;
  VLOG(3) << "start build CPU ps thread.";
  pre_build_threads_ = std::thread([this] { pre_build_thread(); });
}

void PSGPUWrapper::pre_build_thread() {
  // prebuild: process load_data
  while (running_) {
    std::shared_ptr<HeterContext> gpu_task = nullptr;
    if (!data_ready_channel_->Get(gpu_task)) {
      continue;
    }
    VLOG(3) << "thread PreBuildTask start.";
    platform::Timer timer;
    timer.Start();
    // build cpu ps data process
    PreBuildTask(gpu_task);
    timer.Pause();
    VLOG(0) << "thread PreBuildTask end, cost time: " << timer.ElapsedSec()
            << " s";
    buildcpu_ready_channel_->Put(gpu_task);
  }
  VLOG(3) << "build cpu thread end";
}

void PSGPUWrapper::build_task() {
  // build_task: build_pull + build_gputask
  std::shared_ptr<HeterContext> gpu_task = nullptr;
  // train end, gpu free
  if (!gpu_free_channel_->Get(gpu_task)) {
    return;
  }
  // ins and pre_build end
  if (!buildcpu_ready_channel_->Get(gpu_task)) {
    return;
  }

  VLOG(0) << "BuildPull start.";
  platform::Timer timer;
  timer.Start();
  BuildPull(gpu_task);
  BuildGPUTask(gpu_task);
  timer.Pause();
  VLOG(0) << "BuildPull + BuildGPUTask end, cost time: " << timer.ElapsedSec()
          << "s";

  current_task_ = gpu_task;
}

void PSGPUWrapper::BeginPass() {
  platform::Timer timer;
  timer.Start();
  if (current_task_) {
    PADDLE_THROW(
        platform::errors::Fatal("[BeginPass] current task is not ended."));
  }

  build_task();
  timer.Pause();

  if (current_task_ == nullptr) {
    PADDLE_THROW(platform::errors::Fatal(
        "[BeginPass] after build_task, current task is not null."));
  }

  VLOG(0) << "BeginPass end, cost time: " << timer.ElapsedSec() << "s";
}

void PSGPUWrapper::EndPass() {
  if (!current_task_) {
    PADDLE_THROW(
        platform::errors::Fatal("[EndPass] current task has been ended."));
  }
  platform::Timer timer;
  timer.Start();
  size_t keysize_max = 0;
  // in case of feasign_num = 0, skip dump_to_cpu
  for (size_t i = 0; i < heter_devices_.size(); i++) {
    keysize_max = std::max(keysize_max, current_task_->device_keys_[i].size());
  }
  if (keysize_max != 0) {
    HeterPs_->end_pass();
  }

  gpu_task_pool_.Push(current_task_);
  current_task_ = nullptr;
  gpu_free_channel_->Put(current_task_);
  timer.Pause();
  VLOG(0) << "EndPass end, cost time: " << timer.ElapsedSec() << "s";
}

void PSGPUWrapper::PullSparse(const paddle::platform::Place& place,
                              const int table_id,
                              const std::vector<const uint64_t*>& keys,
                              const std::vector<float*>& values,
                              const std::vector<int64_t>& slot_lengths,
                              const int hidden_size) {
  platform::Timer all_timer;
  platform::Timer pull_gpups_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  VLOG(3) << "Begine Gpu/Xpu Ps PullSparse";
  auto buf = memory::Alloc(place, total_length * sizeof(FeatureValue));
  FeatureValue* total_values_gpu = reinterpret_cast<FeatureValue*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GpuPs now."));
  } else if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    LoDTensor& total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys = reinterpret_cast<uint64_t*>(
        total_keys_tensor.mutable_data<int64_t>({total_length, 1}, place));

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
    cudaMemcpy(gpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_len, slot_lengths_lod.data(),
               slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    this->CopyKeys(place, gpu_keys, total_keys, gpu_len,
                   static_cast<int>(slot_lengths.size()),
                   static_cast<int>(total_length));
    VLOG(3) << "Begin call PullSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    pull_gpups_timer.Start();
    HeterPs_->pull_sparse(devid_2_index, total_keys, total_values_gpu,
                          static_cast<int>(total_length));
    // PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
    //                              "PullSparseGPU failed in GPUPS."));
    pull_gpups_timer.Pause();

    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";
    this->CopyForPull(place, gpu_keys, values, total_values_gpu, gpu_len,
                      static_cast<int>(slot_lengths.size()), hidden_size,
                      total_length);
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU_KP
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    LoDTensor& total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys = reinterpret_cast<uint64_t*>(
        total_keys_tensor.mutable_data<int64_t>({total_length, 1}, place));

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
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(xpu_keys, keys.data(),
                                          keys.size() * sizeof(uint64_t*),
                                          XPU_HOST_TO_DEVICE));
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(xpu_len, slot_lengths_lod.data(),
                                          slot_lengths.size() * sizeof(int64_t),
                                          XPU_HOST_TO_DEVICE));

    this->CopyKeys(place, xpu_keys, total_keys, xpu_len,
                   static_cast<int>(slot_lengths.size()),
                   static_cast<int>(total_length));
    VLOG(3) << "Begin call PullSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    pull_gpups_timer.Start();
    HeterPs_->pull_sparse(devid_2_index, total_keys, total_values_gpu,
                          static_cast<int>(total_length));
    pull_gpups_timer.Pause();

    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";
    this->CopyForPull(place, xpu_keys, values, total_values_gpu, xpu_len,
                      static_cast<int>(slot_lengths.size()), hidden_size,
                      total_length);
#endif
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GpuPs/XpuPs: PullSparse Only Support CUDAPlace or XPUPlace Now."));
  }
  all_timer.Pause();
  VLOG(3) << "GpuPs PullSparse total costs: " << all_timer.ElapsedSec()
          << " s, of which GPUPS costs: " << pull_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PullSparse";
}

void PSGPUWrapper::PushSparseGrad(const paddle::platform::Place& place,
                                  const int table_id,
                                  const std::vector<const uint64_t*>& keys,
                                  const std::vector<const float*>& grad_values,
                                  const std::vector<int64_t>& slot_lengths,
                                  const int hidden_size, const int batch_size) {
  platform::Timer all_timer;
  platform::Timer push_gpups_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  // #ifdef PADDLE_WITH_CUDA
  VLOG(3) << "Begin GPUPS PushSparseGrad";
  auto buf = memory::Alloc(place, total_length * sizeof(FeaturePushValue));
  FeaturePushValue* total_grad_values_gpu =
      reinterpret_cast<FeaturePushValue*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GPUPS now."));
  } else if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    LoDTensor& cached_total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys =
        reinterpret_cast<uint64_t*>(cached_total_keys_tensor.data<int64_t>());
    VLOG(3) << "Begin copy grad tensor to gpups struct";
    this->CopyForPush(place, grad_values, total_grad_values_gpu, slot_lengths,
                      hidden_size, total_length, batch_size);

    VLOG(3) << "Begin call PushSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    push_gpups_timer.Start();
    HeterPs_->push_sparse(devid_2_index, total_keys, total_grad_values_gpu,
                          static_cast<int>(total_length));
    push_gpups_timer.Pause();
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU_KP
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    LoDTensor& cached_total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys =
        reinterpret_cast<uint64_t*>(cached_total_keys_tensor.data<int64_t>());
    VLOG(3) << "Begin copy grad tensor to xpups struct";
    this->CopyForPush(place, grad_values, total_grad_values_gpu, slot_lengths,
                      hidden_size, total_length, batch_size);

    VLOG(3) << "Begin call PushSparseXPU in XPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    push_gpups_timer.Start();
    HeterPs_->push_sparse(devid_2_index, total_keys, total_grad_values_gpu,
                          static_cast<int>(total_length));
    push_gpups_timer.Pause();
#endif
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GPUPS: PushSparseGrad Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  VLOG(3) << "PushSparseGrad total cost: " << all_timer.ElapsedSec()
          << " s, of which GPUPS cost: " << push_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PushSparseGrad";
}

}  // end namespace framework
}  // end namespace paddle
#endif
