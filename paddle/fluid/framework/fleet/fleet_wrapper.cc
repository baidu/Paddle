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

#include "paddle/fluid/framework/fleet/fleet_wrapper.h"

#include "glog/logging.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle::framework {

const uint32_t MAX_FEASIGN_NUM = 1024 * 100 * 100;
std::shared_ptr<FleetWrapper> FleetWrapper::s_instance_ = NULL;
bool FleetWrapper::is_initialized_ = false;
std::mutex FleetWrapper::ins_mutex;

#ifdef PADDLE_WITH_PSLIB
std::shared_ptr<paddle::distributed::PSlib> FleetWrapper::pslib_ptr_ = NULL;
#endif

void FleetWrapper::SetClient2ClientConfig(int request_timeout_ms,
                                          int connect_timeout_ms,
                                          int max_retry) {
  client2client_request_timeout_ms_ = request_timeout_ms;
  client2client_connect_timeout_ms_ = connect_timeout_ms;
  client2client_max_retry_ = max_retry;
}

void FleetWrapper::InitServer(const std::string& dist_desc, int index) {
#ifdef PADDLE_WITH_PSLIB
  if (!is_initialized_) {
    VLOG(3) << "Going to init server";
    pslib_ptr_ = std::shared_ptr<paddle::distributed::PSlib>(
        new paddle::distributed::PSlib());
    pslib_ptr_->init_server(dist_desc, index);
    is_initialized_ = true;
  } else {
    VLOG(3) << "Server can be initialized only once";
  }
#endif
}

void FleetWrapper::InitWorker(const std::string& dist_desc,
                              const std::vector<uint64_t>& host_sign_list,
                              int node_num,
                              int index) {
#ifdef PADDLE_WITH_PSLIB
  if (!is_initialized_) {
    VLOG(0) << "Going to init worker";
    pslib_ptr_ = std::shared_ptr<paddle::distributed::PSlib>(
        new paddle::distributed::PSlib());
    pslib_ptr_->init_worker(dist_desc,
                            const_cast<uint64_t*>(host_sign_list.data()),
                            node_num,
                            index);
    dist_desc_ = dist_desc;
    is_initialized_ = true;
  } else {
    VLOG(3) << "Worker can be initialized only once";
  }
#endif
}

void FleetWrapper::StopServer() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to stop server";
  pslib_ptr_->stop_server();
#endif
}

void FleetWrapper::FinalizeWorker() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to finalize worker";
  pslib_ptr_->finalize_worker();
#endif
}

uint64_t FleetWrapper::RunServer() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to run server";
  return pslib_ptr_->run_server();
#else
  return 0;
#endif
}

uint64_t FleetWrapper::RunServer(const std::string& ip, uint32_t port) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to run server with ip " << ip << " port " << port;
  auto ret = pslib_ptr_->run_server(ip, port);
  return ret;
#else
  return 0;
#endif
}

void FleetWrapper::GatherServers(const std::vector<uint64_t>& host_sign_list,
                                 int node_num) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to gather server ips";
  pslib_ptr_->gather_servers(const_cast<uint64_t*>(host_sign_list.data()),
                             node_num);
#endif
}

void FleetWrapper::GatherClients(const std::vector<uint64_t>& host_sign_list) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(0) << "Going to gather client ips";
  size_t len = host_sign_list.size();
  pslib_ptr_->gather_clients(const_cast<uint64_t*>(host_sign_list.data()), len);
#endif
}

std::vector<uint64_t> FleetWrapper::GetClientsInfo() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to get client info";
  return pslib_ptr_->get_client_info();
#endif
  return std::vector<uint64_t>();
}

void FleetWrapper::CreateClient2ClientConnection() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(0) << "Going to create client2client connection";
  pslib_ptr_->create_client2client_connection(client2client_request_timeout_ms_,
                                              client2client_connect_timeout_ms_,
                                              client2client_max_retry_);
#endif
}

#ifdef PADDLE_WITH_PSLIB
void FleetWrapper::HeterPullSparseVars(
    int workerid,
    std::shared_ptr<HeterTask> task,
    const uint64_t table_id,
    const std::vector<std::string>& var_names,
    int fea_value_dim,
    const std::vector<std::string>& var_emb_names) {
  std::vector<::std::future<int32_t>> pull_sparse_status;
  pull_sparse_status.resize(0);
  auto& scope = *(task->scope_);
  auto& fea_keys = (task->features_)[table_id];
  auto& fea_values = (task->feature_values_)[table_id];
  fea_keys.clear();
  for (size_t var_index = 0; var_index < var_names.size(); ++var_index) {
    const std::string& name = var_names[var_index];
    Variable* var = scope.FindVar(name);
    if (var == nullptr) {
      continue;
    }
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        common::errors::InvalidArgument(
            "The tensor for variable '%s' is null. "
            "Ensure that the tensor is properly initialized before use.",
            name.c_str()));

    int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();

    // skip slots which do not have embedding
    const std::string& emb_name = var_emb_names[var_index];
    Variable* emb_var = scope.FindVar(emb_name);
    if (emb_var == nullptr) {
      continue;
    }

    for (auto i = 0u; i < len; ++i) {
      if (ids[i] == 0u) {
        continue;
      }
      fea_keys.push_back(static_cast<uint64_t>(ids[i]));
    }
  }
  fea_values.resize(fea_keys.size() + 1);
  for (auto& t : fea_values) {
    t.resize(fea_value_dim);
  }
  std::vector<float*> pull_result_ptr;
  for (auto& t : fea_values) {
    pull_result_ptr.push_back(t.data());
  }
  auto status =
      pslib_ptr_->_worker_ptr->heter_pull_sparse(workerid,
                                                 pull_result_ptr.data(),
                                                 table_id,
                                                 fea_keys.data(),
                                                 fea_keys.size(),
                                                 task->taskid_);
  pull_sparse_status.push_back(std::move(status));
  for (auto& t : pull_sparse_status) {
    t.wait();
    auto status = t.get();
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(sleep_seconds_before_fail_exit_);
      exit(-1);
    }
  }
}

void FleetWrapper::HeterPushSparseVars(
    std::shared_ptr<HeterTask> task,
    const Scope& scope,
    const uint64_t table_id,
    const std::vector<std::string>& sparse_key_names,
    const std::vector<std::string>& sparse_grad_names,
    const int emb_dim,
    std::vector<::std::future<int32_t>>* push_sparse_status,
    const bool use_cvm,
    const bool dump_slot,
    const bool no_cvm) {
  int batch_size = task->cur_batch_;
  int offset = 2;
  int slot_offset = 0;
  int grad_dim = emb_dim;
  int show_index = 0;
  int click_index = 1;
  auto& fea_keys = (task->features_)[table_id];
  auto& fea_labels = (task->feature_labels_)[table_id];
  auto& push_values = (task->feature_grads_)[table_id];
  auto& sparse_push_keys = (task->sparse_push_keys_)[table_id];

  if (use_cvm) {
    offset = 0;
    grad_dim = emb_dim - 2;
  }
  if (no_cvm) {
    offset = 0;
    grad_dim = emb_dim;
  }
  if (dump_slot) {
    slot_offset = 1;
    show_index = 1;
    click_index = 2;
  }

  PADDLE_ENFORCE_GE(grad_dim,
                    0,
                    common::errors::InvalidArgument(
                        "The gradient dimension (grad_dim) must be greater "
                        "than or equal to 0, but got %d.",
                        grad_dim));

  sparse_push_keys.clear();
  sparse_push_keys.reserve(fea_keys.size() + 1);
  push_values.resize(fea_keys.size() + 1);
  for (auto& t : push_values) {
    t.resize(emb_dim + offset + slot_offset);
  }
  uint64_t fea_idx = 0u;
  for (size_t i = 0;
       i < sparse_key_names.size() && i < sparse_grad_names.size();
       ++i) {
    Variable* var = scope.FindVar(sparse_key_names[i]);
    if (var == nullptr) {
      continue;
    }
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    if (tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    size_t len = tensor->numel();
    int64_t* ids = tensor->data<int64_t>();
    int slot = 0;
    if (dump_slot) {
      slot = std::stoi(sparse_key_names[i]);
    }
    Variable* g_var = scope.FindVar(sparse_grad_names[i]);
    if (g_var == nullptr) {
      continue;
    }
    phi::DenseTensor* g_tensor = g_var->GetMutable<phi::DenseTensor>();
    if (g_tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    float* g = g_tensor->data<float>();

    if (scale_sparse_gradient_with_batch_size_ && grad_dim > 0) {
      int dim = emb_dim + offset;
      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          g_mat(g, g_tensor->numel() / dim, dim);
      g_mat.rightCols(grad_dim) *= batch_size;
    }
    for (auto id_idx = 0u; id_idx < len; ++id_idx) {
      if (ids[id_idx] == 0) {
        g += emb_dim;
        continue;
      }
      sparse_push_keys.push_back(ids[id_idx]);
      PADDLE_ENFORCE_LT(fea_idx,
                        push_values.size(),
                        common::errors::InvalidArgument(
                            "The feature index (fea_idx) must be less than the "
                            "size of push_values. "
                            "Received fea_idx: %d, push_values size: %zu.",
                            fea_idx,
                            push_values.size()));

      if (use_cvm || no_cvm) {
        memcpy(push_values[fea_idx].data() + offset + slot_offset,
               g,
               sizeof(float) * emb_dim);
      } else {
        PADDLE_ENFORCE_LT(fea_idx,
                          fea_labels.size(),
                          common::errors::InvalidArgument(
                              "The feature index (fea_idx) must be less than "
                              "the size of fea_labels. "
                              "Received fea_idx: %d, fea_labels size: %zu.",
                              fea_idx,
                              fea_labels.size()));

        memcpy(push_values[fea_idx].data() + offset + slot_offset,
               g,
               sizeof(float) * emb_dim);
        push_values[fea_idx][show_index] = 1.0f;
        push_values[fea_idx][click_index] =
            static_cast<float>(fea_labels[fea_idx]);
      }
      if (dump_slot) {
        push_values[fea_idx][0] = static_cast<float>(slot);
      }
      g += emb_dim;
      fea_idx++;
    }
  }
  // slots whose embedding has been stop gradient or
  // not involved in forward-backward
  uint64_t no_grad_fea_num = 0u;
  for (size_t i = sparse_grad_names.size(); i < sparse_key_names.size(); ++i) {
    Variable* var = scope.FindVar(sparse_key_names[i]);
    if (var == nullptr) {
      continue;
    }
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    if (tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    size_t len = tensor->numel();
    int64_t* ids = tensor->data<int64_t>();
    for (auto id_idx = 0u; id_idx < len; ++id_idx) {
      if (ids[id_idx] == 0) {
        continue;
      }
      ++no_grad_fea_num;
    }
  }
  PADDLE_ENFORCE_EQ(
      fea_idx + no_grad_fea_num,
      fea_keys.size(),
      common::errors::InvalidArgument(
          "The sum of fea_idx and no_grad_fea_num must be equal to the size of "
          "fea_keys. "
          "Received fea_idx: %d, no_grad_fea_num: %d, fea_keys size: %zu.",
          fea_idx,
          no_grad_fea_num,
          fea_keys.size()));

  PADDLE_ENFORCE_EQ(
      fea_idx,
      sparse_push_keys.size(),
      common::errors::InvalidArgument(
          "The fea_idx must be equal to the size of sparse_push_keys. "
          "Received fea_idx: %d, sparse_push_keys size: %zu.",
          fea_idx,
          sparse_push_keys.size()));

  if (fea_idx == 0) {
    return;
  }
  std::vector<float*> push_g_vec;
  for (auto i = 0u; i < sparse_push_keys.size(); ++i) {
    push_g_vec.push_back(push_values[i].data());
  }
  auto status =
      pslib_ptr_->_worker_ptr->push_sparse(table_id,
                                           sparse_push_keys.data(),
                                           (const float**)push_g_vec.data(),
                                           sparse_push_keys.size());
  push_sparse_status->push_back(std::move(status));
}
#endif

int FleetWrapper::RegisterHeterCallback(HeterCallBackFunc handler) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "calling FleetWrapper::RegisterHeterCallback";
  VLOG(3) << "pslib_ptr_=" << pslib_ptr_;
  VLOG(3) << "_worker_ptr=" << pslib_ptr_->_worker_ptr;
  return pslib_ptr_->_worker_ptr->register_heter_callback(handler);

#else
  VLOG(0) << "FleetWrapper::RegisterHeterCallback"
          << " does nothing when no pslib";
#endif
  return 0;
}

void FleetWrapper::PullSparseToLocal(const uint64_t table_id,
                                     int fea_value_dim) {
#ifdef PADDLE_WITH_PSLIB
  size_t fea_keys_size = local_tables_.size();
  if (fea_keys_size == 0) {
    return;
  }
  local_table_shard_num_ = fea_keys_size;
  platform::Timer timeline;
  std::vector<std::thread> threads(fea_keys_size);
  auto ptl_func = [this, &table_id](int i) {
    size_t key_size = this->local_tables_[i].size();
    std::vector<uint64_t> keys;
    keys.reserve(key_size);
    std::vector<float*> pull_result_ptr;
    pull_result_ptr.reserve(key_size);

    for (auto& kv : this->local_tables_[i]) {
      keys.emplace_back(kv.first);
      pull_result_ptr.emplace_back(kv.second.data());
    }
    auto tt = pslib_ptr_->_worker_ptr->pull_sparse(
        pull_result_ptr.data(), table_id, keys.data(), key_size);
    tt.wait();
    auto status = tt.get();
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(sleep_seconds_before_fail_exit_);
      exit(-1);
    } else {
      VLOG(3) << "FleetWrapper Pull sparse to local done with table size: "
              << pull_result_ptr.size();
    }
  };
  timeline.Start();
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(ptl_func, i);
  }
  for (std::thread& t : threads) {
    t.join();
  }
  local_pull_pool_.reset(new ::ThreadPool(pull_local_thread_num_));
  timeline.Pause();
#endif
}

void FleetWrapper::PullSparseVarsFromLocal(
    const Scope& scope,
    const uint64_t table_id,
    const std::vector<std::string>& var_names,
    std::vector<uint64_t>* fea_keys,
    std::vector<std::vector<float>>* fea_values,
    int fea_value_dim) {
#ifdef PADDLE_WITH_PSLIB
  fea_keys->clear();
  fea_keys->resize(0);
  fea_keys->reserve(MAX_FEASIGN_NUM);
  for (auto name : var_names) {
    Variable* var = scope.FindVar(name);
    if (var == nullptr) {
      continue;
    }
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        common::errors::InvalidArgument(
            "The tensor for variable '%s' is null. Please ensure the tensor is "
            "properly initialized before use.",
            name.c_str()));

    int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    for (auto i = 0u; i < len; ++i) {
      if (ids[i] == 0u) {
        continue;
      }
      fea_keys->push_back(static_cast<uint64_t>(ids[i]));
    }
  }
  fea_values->resize(fea_keys->size() + 1);
  for (auto& t : *fea_values) {
    t.resize(fea_value_dim);
  }
  size_t key_length = fea_keys->size();
  int local_step = key_length / pull_local_thread_num_;
  std::vector<std::future<void>> task_futures;
  task_futures.reserve(key_length / local_step + 1);
  for (size_t i = 0; i < key_length; i += local_step) {
    size_t end = i + local_step < key_length ? i + local_step : key_length;
    auto pull_local_task =
        [this, i, end, &fea_values, &fea_keys, &fea_value_dim] {
          for (size_t j = i; j < end; j++) {
            std::memcpy((*fea_values)[j].data(),
                        local_tables_[(*fea_keys)[j] % local_table_shard_num_]
                                     [(*fea_keys)[j]]
                                         .data(),
                        fea_value_dim * sizeof(float));
          }
        };
    task_futures.emplace_back(
        local_pull_pool_->enqueue(std::move(pull_local_task)));
  }
  for (auto& tf : task_futures) {
    tf.wait();
  }
#endif
}

void FleetWrapper::ClearLocalTable() {
#ifdef PADDLE_WITH_PSLIB
  for (auto& t : local_tables_) {
    t.clear();
  }
#endif
}

std::future<int32_t> FleetWrapper::PullSparseVarsAsync(
    const Scope& scope,
    const uint64_t table_id,
    const std::vector<std::string>& var_names,
    std::vector<uint64_t>* fea_keys,
    std::vector<std::vector<float>>* fea_values,
    int fea_value_dim) {
#ifdef PADDLE_WITH_PSLIB
  fea_keys->clear();
  fea_keys->resize(0);
  fea_keys->reserve(MAX_FEASIGN_NUM);
  for (auto name : var_names) {
    Variable* var = scope.FindVar(name);
    if (var == nullptr) {
      continue;
    }
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        common::errors::InvalidArgument(
            "The tensor for variable '%s' is null. Please ensure the tensor is "
            "properly initialized before use.",
            name.c_str()));

    int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    for (auto i = 0u; i < len; ++i) {
      if (ids[i] == 0u) {
        continue;
      }
      fea_keys->push_back(static_cast<uint64_t>(ids[i]));
    }
  }
  fea_values->resize(fea_keys->size() + 1);
  for (auto& t : *fea_values) {
    t.resize(fea_value_dim);
  }
  std::vector<float*> pull_result_ptr;
  for (auto& t : *fea_values) {
    pull_result_ptr.push_back(t.data());
  }
  return pslib_ptr_->_worker_ptr->pull_sparse(
      pull_result_ptr.data(), table_id, fea_keys->data(), fea_keys->size());
#endif
  return std::future<int32_t>();
}

void FleetWrapper::PullSparseVarsSync(
    const Scope& scope,
    const uint64_t table_id,
    const std::vector<std::string>& var_names,
    std::vector<uint64_t>* fea_keys,
    std::vector<std::vector<float>>* fea_values,
    int fea_value_dim,
    const std::vector<std::string>& var_emb_names) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<::std::future<int32_t>> pull_sparse_status;
  pull_sparse_status.resize(0);
  fea_keys->clear();
  fea_keys->resize(0);
  fea_keys->reserve(MAX_FEASIGN_NUM);
  for (size_t var_index = 0; var_index < var_names.size(); ++var_index) {
    const std::string& name = var_names[var_index];
    Variable* var = scope.FindVar(name);
    if (var == nullptr) {
      continue;
    }
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        common::errors::InvalidArgument(
            "The tensor for variable '%s' is null. Please ensure the tensor is "
            "properly initialized before use.",
            name.c_str()));

    int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();

    // skip slots which do not have embedding
    const std::string& emb_name = var_emb_names[var_index];
    Variable* emb_var = scope.FindVar(emb_name);
    if (emb_var == nullptr) {
      continue;
    }

    for (auto i = 0u; i < len; ++i) {
      if (ids[i] == 0u) {
        continue;
      }
      fea_keys->push_back(static_cast<uint64_t>(ids[i]));
    }
  }
  fea_values->resize(fea_keys->size() + 1);
  for (auto& t : *fea_values) {
    t.resize(fea_value_dim);
  }
  std::vector<float*> pull_result_ptr;
  for (auto& t : *fea_values) {
    pull_result_ptr.push_back(t.data());
  }

  int32_t cnt = 0;
  while (true) {
    pull_sparse_status.clear();
    auto status = pslib_ptr_->_worker_ptr->pull_sparse(
        pull_result_ptr.data(), table_id, fea_keys->data(), fea_keys->size());
    pull_sparse_status.push_back(std::move(status));
    bool flag = true;
    for (auto& t : pull_sparse_status) {
      t.wait();
      int32_t status = -1;
      try {
        status = t.get();
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
    }
    if (flag) {
      break;
    }
  }
#endif
}

void FleetWrapper::PullSparseToTensorSync(
    const uint64_t table_id,
    int fea_dim,
    uint64_t padding_id,
    phi::Place place,
    std::vector<const phi::DenseTensor*>* inputs,
    std::vector<phi::DenseTensor*>* outputs) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<uint64_t> fea_keys;
  std::vector<float*> pull_result_ptr;
  fea_keys.reserve(MAX_FEASIGN_NUM / 100);
  pull_result_ptr.reserve(MAX_FEASIGN_NUM / 100);
  std::vector<float> init_value(fea_dim, 0);
  phi::DenseTensor* output = nullptr;
  float* output_data = nullptr;
  size_t output_index = -1;
  size_t output_len = 0;
  for (size_t index = 0; index < inputs->size(); ++index) {
    const phi::DenseTensor* tensor = inputs->at(index);
    const int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    for (size_t i = 0; i < len; ++i, output_len += fea_dim) {
      if (!output || output_len == size_t(output->numel())) {
        ++output_index;
        PADDLE_ENFORCE_LT(
            output_index,
            outputs->size(),
            common::errors::InvalidArgument(
                "The output index must be less than the size of outputs. "
                "Received output_index: %d, outputs size: %zu.",
                output_index,
                outputs->size()));  // NOLINT
        output = outputs->at(output_index);
        output_data = output->mutable_data<float>(place);
        output_len = 0;
        PADDLE_ENFORCE_EQ(output->numel() % fea_dim,
                          0,
                          common::errors::InvalidArgument(
                              "The number of elements in the output must be "
                              "divisible by fea_dim. "
                              "Received output numel: %d, fea_dim: %d.",
                              output->numel(),
                              fea_dim));  // NOLINT
        PADDLE_ENFORCE_NOT_NULL(
            output_data,
            common::errors::InvalidArgument(
                "The output data pointer is null."));  // NOLINT
      }

      uint64_t real_id = static_cast<uint64_t>(ids[i]);
      if (real_id == padding_id) {
        memcpy(output_data + output_len,
               init_value.data(),
               sizeof(float) * fea_dim);
        continue;
      }
      fea_keys.push_back(real_id);
      pull_result_ptr.push_back(output_data + output_len);
    }
  }
  auto status = pslib_ptr_->_worker_ptr->pull_sparse(
      pull_result_ptr.data(), table_id, fea_keys.data(), fea_keys.size());
  status.wait();
  auto ret = status.get();
  if (ret != 0) {
    LOG(ERROR) << "fleet pull sparse failed, status[" << ret << "]";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
#else
  for (size_t index = 0; index < inputs->size(); ++index) {
    auto* tensor = inputs->at(index);
    size_t len = tensor->numel();
    std::vector<float> init_data(fea_dim, 0);
    for (size_t i = 0; i < len; ++i) {
      memcpy(outputs->at(index)->mutable_data<float>(place),
             init_data.data(),
             fea_dim);
    }
  }
#endif
}

void FleetWrapper::PullDenseVarsAsync(
    const Scope& scope,
    const uint64_t tid,
    const std::vector<std::string>& var_names,
    std::vector<::std::future<int32_t>>* pull_dense_status,
    bool in_cpu) {
#ifdef PADDLE_WITH_PSLIB
  auto& regions = _regions[tid];
  regions.clear();
  regions.resize(var_names.size());
  for (auto i = 0u; i < var_names.size(); ++i) {
    std::string varname = var_names[i];
    if (!in_cpu) {
      varname = var_names[i] + "pin";
    }
    Variable* var = scope.FindVar(varname);
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    float* w = tensor->data<float>();
    paddle::ps::Region reg(w, tensor->numel());
    regions[i] = std::move(reg);
  }
  auto status =
      pslib_ptr_->_worker_ptr->pull_dense(regions.data(), regions.size(), tid);
  pull_dense_status->push_back(std::move(status));
#endif
}

void FleetWrapper::PullDenseVarsSync(
    const Scope& scope,
    const uint64_t tid,
    const std::vector<std::string>& var_names) {
#ifdef PADDLE_WITH_PSLIB
  auto& regions = _regions[tid];
  regions.clear();
  regions.reserve(var_names.size());
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    float* w = tensor->data<float>();
    paddle::ps::Region reg(w, tensor->numel());
    regions.emplace_back(std::move(reg));
  }
  int32_t status = -1;
  int32_t cnt = 0;
  while (true) {
    auto tt = pslib_ptr_->_worker_ptr->pull_dense(
        regions.data(), regions.size(), tid);
    bool flag = true;

    tt.wait();

    try {
      status = tt.get();
    } catch (const std::future_error& e) {
      VLOG(0) << "Caught a future_error with code" << e.code()
              << ", Message:" << e.what();
    }
    if (status != 0) {
      VLOG(0) << "fleet pull dense sync failed, status[" << status << "]";
      sleep(sleep_seconds_before_fail_exit_);
      flag = false;
      cnt++;
    }
    if (cnt > 3) {
      VLOG(0) << "fleet pull dense sync failed, retry 3 times";
      exit(-1);
    }

    if (flag) {
      break;
    }
  }
#endif
}

void FleetWrapper::PushDenseParamSync(
    const Scope& scope,
    const uint64_t table_id,
    const std::vector<std::string>& var_names) {
#ifdef PADDLE_WITH_PSLIB
  auto place = phi::CPUPlace();
  std::vector<paddle::ps::Region> regions;
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::NotFound(
            "Variable 'var' with identifier [%s] not found. Please ensure the "
            "variable is correctly initialized.",
            t.c_str()));

    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    float* g = tensor->mutable_data<float>(place);
    paddle::ps::Region reg(g, tensor->numel());
    regions.emplace_back(std::move(reg));
  }
  auto push_status = pslib_ptr_->_worker_ptr->push_dense_param(
      regions.data(), regions.size(), table_id);
  push_status.wait();
  auto status = push_status.get();
  PADDLE_ENFORCE_EQ(
      status,
      0,
      common::errors::InvalidArgument(
          "Pushing dense parameter failed. Received status: %d. Please check "
          "the status and ensure the operation is successful.",
          status));

#endif
}

void FleetWrapper::PushDenseVarsSync(
    Scope* scope,
    const uint64_t table_id,
    const std::vector<std::string>& var_names) {}

#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && \
    (defined PADDLE_WITH_PSLIB)
void FleetWrapper::PushDenseVarsAsync(
    const Scope& scope,
    const uint64_t table_id,
    const std::vector<std::string>& var_names,
    std::vector<::std::future<int32_t>>* push_sparse_status,
    float scale_datanorm,
    int batch_size,
    const phi::Place& place,
    gpuStream_t stream,
    gpuEvent_t event) {
  std::vector<paddle::ps::Region> regions;
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    int count = tensor->numel();
    float* g_data = tensor->data<float>();

    Variable* pin_var = scope.FindVar(t + "pin");
    phi::DenseTensor* pin_tensor = pin_var->GetMutable<phi::DenseTensor>();
    float* pin_g =
        pin_tensor->mutable_data<float>(tensor->dims(), phi::GPUPinnedPlace());
    memory::Copy(phi::GPUPinnedPlace(),
                 pin_g,
                 place,
                 g_data,
                 sizeof(float) * count,
                 stream);
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(event, stream));
    hipEventSynchronize(event);
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, stream));
    cudaEventSynchronize(event);
#endif

    float* g = pin_g;
    if (scale_datanorm >= 0) {
      if (t.find(".batch_size@GRAD") != std::string::npos ||
          t.find(".batch_sum@GRAD") != std::string::npos) {
        Eigen::Map<Eigen::MatrixXf> mat(g, 1, count);
        float scale = 1.0 / batch_size;
        mat *= scale;
      } else if (t.find(".batch_square_sum@GRAD") != std::string::npos) {
        VLOG(3) << "epsilon: " << scale_datanorm;
        for (int i = 0; i < count; ++i) {
          g[i] = (g[i] - batch_size * scale_datanorm) / batch_size +
                 batch_size * scale_datanorm;
        }
      }
    }
    paddle::ps::Region reg(g, count);
    regions.emplace_back(std::move(reg));
  }

  auto status = pslib_ptr_->_worker_ptr->push_dense(
      regions.data(), regions.size(), table_id);
  if (push_sparse_status) {
    push_sparse_status->push_back(std::move(status));
  }
}
#endif

#ifdef PADDLE_WITH_XPU
void FleetWrapper::PushDenseVarsAsync(
    const Scope& scope,
    const uint64_t table_id,
    const std::vector<std::string>& var_names,
    std::vector<::std::future<int32_t>>* push_sparse_status,
    float scale_datanorm,
    int batch_size,
    const phi::Place& place) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<paddle::ps::Region> regions;
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    int count = tensor->numel();
    float* g_data = tensor->data<float>();

    Variable* pin_var = scope.FindVar(t + "pin");
    phi::DenseTensor* pin_tensor = pin_var->GetMutable<phi::DenseTensor>();
    float* pin_g =
        pin_tensor->mutable_data<float>(tensor->dims(), phi::CPUPlace());
    memory::Copy(phi::CPUPlace(), pin_g, place, g_data, sizeof(float) * count);

    float* g = pin_g;
    if (scale_datanorm >= 0) {
      if (t.find(".batch_size@GRAD") != std::string::npos ||
          t.find(".batch_sum@GRAD") != std::string::npos) {
        Eigen::Map<Eigen::MatrixXf> mat(g, 1, count);
        float scale = 1.0 / batch_size;
        mat *= scale;
      } else if (t.find(".batch_square_sum@GRAD") != std::string::npos) {
        VLOG(3) << "epsilon: " << scale_datanorm;
        for (int i = 0; i < count; ++i) {
          g[i] = (g[i] - batch_size * scale_datanorm) / batch_size +
                 batch_size * scale_datanorm;
        }
      }
    }
    paddle::ps::Region reg(g, count);
    regions.emplace_back(std::move(reg));
  }

  auto status = pslib_ptr_->_worker_ptr->push_dense(
      regions.data(), regions.size(), table_id);
  if (push_sparse_status) {
    push_sparse_status->push_back(std::move(status));
  }
#endif
}
#endif
void FleetWrapper::PushDenseVarsAsync(
    const Scope& scope,
    const uint64_t table_id,
    const std::vector<std::string>& var_names,
    std::vector<::std::future<int32_t>>* push_sparse_status,
    float scale_datanorm,
    int batch_size) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<paddle::ps::Region> regions;
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    int count = tensor->numel();
    float* g = tensor->data<float>();
    if (scale_datanorm >= 0) {
      if (t.find(".batch_size@GRAD") != std::string::npos ||
          t.find(".batch_sum@GRAD") != std::string::npos) {
        Eigen::Map<Eigen::MatrixXf> mat(g, 1, count);
        float scale = 1.0 / batch_size;
        mat *= scale;
      } else if (t.find(".batch_square_sum@GRAD") != std::string::npos) {
        VLOG(3) << "epsilon: " << scale_datanorm;
        for (int i = 0; i < count; ++i) {
          g[i] = (g[i] - batch_size * scale_datanorm) / batch_size +
                 batch_size * scale_datanorm;
        }
      }
    }
    paddle::ps::Region reg(g, count);
    regions.emplace_back(std::move(reg));
  }

  auto status = pslib_ptr_->_worker_ptr->push_dense(
      regions.data(), regions.size(), table_id);
  if (push_sparse_status) {
    push_sparse_status->push_back(std::move(status));
  }
#endif
}

void FleetWrapper::PushSparseVarsWithLabelAsync(
    const Scope& scope,
    const uint64_t table_id,
    const std::vector<uint64_t>& fea_keys,
    const std::vector<float>& fea_labels,
    const std::vector<std::string>& sparse_key_names,
    const std::vector<std::string>& sparse_grad_names,
    const int emb_dim,
    std::vector<std::vector<float>>* push_values,
    std::vector<::std::future<int32_t>>* push_sparse_status,
    const int batch_size,
    const bool use_cvm,
    const bool dump_slot,
    std::vector<uint64_t>* sparse_push_keys,
    const bool no_cvm,
    const bool scale_sparse_gradient_with_batch_size) {
#ifdef PADDLE_WITH_PSLIB
  int offset = 2;
  int slot_offset = 0;
  int grad_dim = emb_dim;
  int show_index = 0;
  int click_index = 1;
  if (use_cvm) {
    offset = 0;
    grad_dim = emb_dim - 2;
  }
  if (no_cvm) {
    offset = 0;
    grad_dim = emb_dim;
  }
  if (dump_slot) {
    slot_offset = 1;
    show_index = 1;
    click_index = 2;
  }
  PADDLE_ENFORCE_GE(grad_dim,
                    0,
                    common::errors::InvalidArgument(
                        "The gradient dimension (grad_dim) must be "
                        "greater than or equal to 0. "
                        "Received grad_dim: %d. Please ensure that "
                        "the gradient dimension is valid.",
                        grad_dim));

  sparse_push_keys->clear();
  sparse_push_keys->reserve(fea_keys.size() + 1);
  push_values->resize(fea_keys.size() + 1);
  for (auto& t : *push_values) {
    t.resize(emb_dim + offset + slot_offset);
  }
  uint64_t fea_idx = 0u;
  for (size_t i = 0;
       i < sparse_key_names.size() && i < sparse_grad_names.size();
       ++i) {
    Variable* var = scope.FindVar(sparse_key_names[i]);
    if (var == nullptr) {
      continue;
    }
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    if (tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    size_t len = tensor->numel();
    int64_t* ids = tensor->data<int64_t>();
    int slot = 0;
    if (dump_slot) {
      try {
        slot = std::stoi(sparse_key_names[i]);
      } catch (std::invalid_argument const& e) {
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "sparse var's name: %s, doesn't support non-integer type name when "
            "dump_slot=True",
            sparse_key_names[i]));
      } catch (std::out_of_range const& e) {
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "sparse var's name: %s, integer type name out of range when "
            "dump_slot=True",
            sparse_key_names[i]));
      }
    }
    Variable* g_var = scope.FindVar(sparse_grad_names[i]);
    if (g_var == nullptr) {
      continue;
    }
    phi::DenseTensor* g_tensor = g_var->GetMutable<phi::DenseTensor>();
    if (g_tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    float* g = g_tensor->data<float>();

    if (scale_sparse_gradient_with_batch_size && grad_dim > 0) {
      int dim = emb_dim;
      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          g_mat(g, g_tensor->numel() / dim, dim);
      g_mat.rightCols(grad_dim) *= batch_size;
    }
    for (auto id_idx = 0u; id_idx < len; ++id_idx) {
      if (ids[id_idx] == 0) {
        g += emb_dim;
        continue;
      }
      sparse_push_keys->push_back(ids[id_idx]);
      PADDLE_ENFORCE_LT(
          fea_idx,
          (*push_values).size(),
          common::errors::InvalidArgument(
              "Feature index (fea_idx) must be less than the size of "
              "push_values. "
              "Received fea_idx: %d, push_values size: %zu. Please ensure the "
              "feature index is within the valid range.",
              fea_idx,
              (*push_values).size()));

      if (use_cvm || no_cvm) {
        memcpy((*push_values)[fea_idx].data() + offset + slot_offset,
               g,
               sizeof(float) * emb_dim);
      } else {
        PADDLE_ENFORCE_LT(
            fea_idx,
            fea_labels.size(),
            common::errors::InvalidArgument(
                "Feature index (fea_idx) must be less than the size of "
                "fea_labels. "
                "Received fea_idx: %d, fea_labels size: %zu. Please ensure the "
                "feature index is within the valid range.",
                fea_idx,
                fea_labels.size()));

        memcpy((*push_values)[fea_idx].data() + offset + slot_offset,
               g,
               sizeof(float) * emb_dim);
        (*push_values)[fea_idx][show_index] = 1.0f;
        (*push_values)[fea_idx][click_index] =
            static_cast<float>(fea_labels[fea_idx]);
      }
      if (dump_slot) {
        (*push_values)[fea_idx][0] = static_cast<float>(slot);
      }
      g += emb_dim;
      fea_idx++;
    }
  }
  // slots whose embedding has been stop gradient or
  // not involved in forward-backward
  uint64_t no_grad_fea_num = 0u;
  for (size_t i = sparse_grad_names.size(); i < sparse_key_names.size(); ++i) {
    Variable* var = scope.FindVar(sparse_key_names[i]);
    if (var == nullptr) {
      continue;
    }
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    if (tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    size_t len = tensor->numel();
    int64_t* ids = tensor->data<int64_t>();
    for (auto id_idx = 0u; id_idx < len; ++id_idx) {
      if (ids[id_idx] == 0) {
        continue;
      }
      ++no_grad_fea_num;
    }
  }
  PADDLE_ENFORCE_EQ(
      fea_idx + no_grad_fea_num,
      fea_keys.size(),
      common::errors::InvalidArgument(
          "The sum of feature index (fea_idx) and no_grad_fea_num must be "
          "equal to the size of fea_keys. "
          "Received fea_idx: %d, no_grad_fea_num: %d, fea_keys size: %zu.",
          fea_idx,
          no_grad_fea_num,
          fea_keys.size()));

  PADDLE_ENFORCE_EQ(fea_idx,
                    sparse_push_keys->size(),
                    common::errors::InvalidArgument(
                        "The feature index (fea_idx) must be equal to the size "
                        "of sparse_push_keys. "
                        "Received fea_idx: %d, sparse_push_keys size: %zu.",
                        fea_idx,
                        sparse_push_keys->size()));

  if (fea_idx == 0) {
    return;
  }
  std::vector<float*> push_g_vec;
  for (auto i = 0u; i < sparse_push_keys->size(); ++i) {
    push_g_vec.push_back((*push_values)[i].data());
  }
  auto status =
      pslib_ptr_->_worker_ptr->push_sparse(table_id,
                                           sparse_push_keys->data(),
                                           (const float**)push_g_vec.data(),
                                           sparse_push_keys->size());
  push_sparse_status->push_back(std::move(status));
#endif
}

void FleetWrapper::PushSparseFromTensorWithLabelAsync(
    const Scope& scope,
    const uint64_t table_id,
    int fea_dim,
    uint64_t padding_id,
    bool scale_sparse,
    const std::string& accessor,
    const std::string& click_name,
    phi::Place place,
    const std::vector<std::string>& input_names,
    std::vector<const phi::DenseTensor*>* inputs,
    std::vector<const phi::DenseTensor*>* outputs) {
#ifdef PADDLE_WITH_PSLIB
  int show_index = 0;
  int click_index = 1;
  // these default values can not be used, it must be set.
  bool dump_slot = false;
  int slot_offset = 0;
  int grad_dim = 0;
  // don't worry, user do not have to care about all these flags
  if (accessor == "DownpourCtrAccessor" ||
      accessor == "DownpourCtrDymfAccessor") {
    dump_slot = true;
    slot_offset = 1;
    grad_dim = fea_dim - 2;
    show_index = 1;
    click_index = 2;
  } else if (accessor == "DownpourFeatureValueAccessor") {
    dump_slot = false;
    slot_offset = 0;
    grad_dim = fea_dim - 2;
  } else if (accessor == "DownpourSparseValueAccessor") {
    dump_slot = false;
    slot_offset = 0;
    grad_dim = fea_dim;
  }
  PADDLE_ENFORCE_GE(grad_dim,
                    0,
                    common::errors::InvalidArgument(
                        "The gradient dimension (grad_dim) must be "
                        "greater than or equal to 0. "
                        "Received grad_dim: %d. Please ensure that "
                        "the gradient dimension is valid.",
                        grad_dim));

  int batch_size = -1;
  for (auto* input : *inputs) {
    int cur_batch_size =
        input->lod().size() ? input->lod()[0].size() - 1 : input->dims()[0];
    if (batch_size == -1) {
      batch_size = cur_batch_size;
    } else {
      PADDLE_ENFORCE_EQ(batch_size,
                        cur_batch_size,
                        common::errors::InvalidArgument(
                            "The batch size (batch_size) must be equal to the "
                            "current batch size (cur_batch_size). "
                            "Received batch_size: %d, cur_batch_size: %d. "
                            "Please ensure that the batch sizes match.",
                            batch_size,
                            cur_batch_size));
      // NOLINT
    }
  }
  PADDLE_ENFORCE_GT(batch_size,
                    0,
                    common::errors::InvalidArgument(
                        "The batch size (batch_size) must be greater than 0. "
                        "Received batch_size: %d. Please ensure that the batch "
                        "size is valid.",
                        batch_size));
  // NOLINT

  std::vector<float> g;
  for (const phi::DenseTensor* g_tensor : *outputs) {
    size_t origin = g.size();
    size_t add = g_tensor->numel();
    g.resize(origin + add);
    memcpy(g.data() + origin, g_tensor->data<float>(), add);
  }
  if (scale_sparse && grad_dim > 0) {
    size_t dim = static_cast<size_t>(grad_dim);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        g_mat(g.data(), g.size() / dim, dim);
    g_mat.rightCols(grad_dim) *= batch_size;
  }

  std::vector<float> fea_labels;
  fea_labels.reserve(MAX_FEASIGN_NUM / 100);
  framework::Variable* var = scope.FindVar(click_name);
  size_t global_idx = 0;
  if (click_name != "") {
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::InvalidArgument("The variable (var) is null when "
                                        "click_name is not an empty string."));

    phi::DenseTensor* label_tensor = var->GetMutable<phi::DenseTensor>();
    PADDLE_ENFORCE_NOT_NULL(label_tensor,
                            common::errors::InvalidArgument(
                                "The label tensor is null when attempting "
                                "to get mutable DenseTensor from var."));

    int64_t* label_ptr = label_tensor->data<int64_t>();

    for (auto* tensor : *inputs) {
      const int64_t* ids = tensor->data<int64_t>();
      size_t fea_idx = 0;
      for (size_t lod_idx = 1; lod_idx < tensor->lod()[0].size(); ++lod_idx) {
        size_t cur = GetAbsoluteSum(tensor->lod()[0][lod_idx - 1],
                                    tensor->lod()[0][lod_idx],
                                    0,
                                    tensor->lod());
        for (size_t i = 0; i < cur; ++i, ++fea_idx) {
          if (static_cast<uint64_t>(ids[fea_idx]) == padding_id) {
            continue;
          }
          fea_labels.push_back(static_cast<float>(label_ptr[lod_idx - 1]));
          ++global_idx;
        }
      }
    }
  }
  std::vector<uint64_t> push_keys;
  push_keys.reserve(MAX_FEASIGN_NUM / 100);
  std::vector<std::vector<float>> push_values;
  push_values.reserve(MAX_FEASIGN_NUM / 100);
  size_t output_len = 0;
  size_t input_idx = 0;
  for (size_t index = 0; index < inputs->size(); ++index) {
    const phi::DenseTensor* tensor = inputs->at(index);
    const int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    for (size_t i = 0; i < len; ++i, output_len += fea_dim) {
      if (static_cast<uint64_t>(ids[i]) == padding_id) {
        continue;
      }
      push_keys.emplace_back(ids[i]);
      push_values.emplace_back(fea_dim + slot_offset);
      float* data = push_values.back().data();
      if (!var) {
        memcpy(
            data + slot_offset, g.data() + output_len, sizeof(float) * fea_dim);
      } else {
        memcpy(data + slot_offset,
               g.data() + output_len,
               sizeof(float) * grad_dim);
        data[show_index] = 1.0f;
        data[click_index] = static_cast<float>(fea_labels.at(input_idx));
      }
      if (dump_slot) {
        int slot = std::stoi(input_names[index]);
        data[0] = static_cast<float>(slot);
      }
      ++input_idx;
    }
  }

  PADDLE_ENFORCE_EQ(
      output_len,
      g.size(),
      common::errors::InvalidArgument(
          "The output length (output_len) must be equal to the size of g. "
          "Received output_len: %d, g.size(): %d.",
          output_len,
          g.size()));

  if (click_name != "") {
    PADDLE_ENFORCE_EQ(
        input_idx,
        global_idx,
        common::errors::InvalidArgument(
            "The input index (input_idx) must be equal to the global index "
            "(global_idx) when click_name is not an empty string. "
            "Received input_idx: %d, global_idx: %d.",
            input_idx,
            global_idx));
  }

  std::vector<float*> push_g_vec(input_idx, nullptr);
  for (auto i = 0u; i < push_keys.size(); ++i) {
    push_g_vec[i] = push_values.at(i).data();
  }
  auto status =
      pslib_ptr_->_worker_ptr->push_sparse(table_id,
                                           push_keys.data(),
                                           (const float**)push_g_vec.data(),
                                           push_keys.size());
#endif
}

void FleetWrapper::LoadFromPaddleModel(Scope& scope,
                                       const uint64_t table_id,
                                       std::vector<std::string> var_list,
                                       std::string model_path,
                                       std::string model_proto_file,
                                       std::vector<std::string> table_var_list,
                                       bool load_combine) {
#ifdef PADDLE_WITH_PSLIB
  // load ProgramDesc from model file
  auto read_proto_func = [](const std::string& filename) -> ProgramDesc {
    std::string contents;
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    fin.seekg(0, std::ios::end);
    contents.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&contents[0], contents.size());
    fin.close();
    ProgramDesc program_desc(contents);
    return program_desc;
  };
  const ProgramDesc old_program = read_proto_func(model_proto_file);
  Scope* old_scope = new Scope();
  auto& old_block = old_program.Block(0);
  auto place = phi::CPUPlace();
  std::vector<std::string> old_param_list;

  for (auto& t : var_list) {
    VarDesc* old_var_desc = old_block.FindVar(t);
    if (old_var_desc == nullptr) {
      continue;
    }
    // init variable in scope
    Variable* old_var = old_scope->Var(old_var_desc->Name());
    InitializeVariable(old_var, old_var_desc->GetType());
    old_param_list.push_back(t);
    if (load_combine) {
      continue;
    }
    // load variable from model
    paddle::framework::AttributeMap attrs;
    attrs.insert({"file_path", model_path + "/" + old_var_desc->Name()});
    auto load_op = paddle::framework::OpRegistry::CreateOp(
        "load", {}, {{"Out", {old_var_desc->Name()}}}, attrs);
    load_op->Run(*old_scope, place);
  }

  if (load_combine) {
    std::sort(old_param_list.begin(), old_param_list.end());
    paddle::framework::AttributeMap attrs;
    attrs.insert({"file_path", model_path});
    auto load_op = paddle::framework::OpRegistry::CreateOp(
        "load_combine", {}, {{"Out", old_param_list}}, attrs);
    load_op->Run(*old_scope, place);
  }

  for (auto& t : old_param_list) {
    Variable* old_var = old_scope->Var(t);
    // old model data, here we assume data type is float
    phi::DenseTensor* old_tensor = old_var->GetMutable<phi::DenseTensor>();
    float* old_data = old_tensor->data<float>();
    // new model data, here we assume data type is float
    Variable* var = scope.FindVar(t);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::NotFound("Variable (var) is null. var[%s] not found.",
                                 t));

    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    float* data = tensor->data<float>();
    // copy from old data to new data
    if (old_tensor->numel() > tensor->numel()) {
      memcpy(data, old_data, tensor->numel() * sizeof(float));
    } else {
      memcpy(data, old_data, old_tensor->numel() * sizeof(float));
    }
  }
  delete old_scope;
  PushDenseParamSync(scope, table_id, table_var_list);
#endif
}

void FleetWrapper::LoadModel(const std::string& path, const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->load(path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "load model from path:" << path << " failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::LoadModel does nothing when no pslib";
#endif
}

void FleetWrapper::LoadModelOneTable(const uint64_t table_id,
                                     const std::string& path,
                                     const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret =
      pslib_ptr_->_worker_ptr->load(table_id, path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "load model of table id: " << table_id
               << ", from path: " << path << " failed";
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::LoadModel does nothing when no pslib";
#endif
}

void FleetWrapper::LoadWithWhitelist(const uint64_t table_id,
                                     const std::string& path,
                                     const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->load_with_whitelist(
      table_id, path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "load model of table id: " << table_id
               << ", from path: " << path << " failed";
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::LoadWhitelist does nothing when no pslib";
#endif
}

void FleetWrapper::SaveMultiTableOnePath(const std::vector<int>& table_ids,
                                         const std::string& path,
                                         const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->save_multi_table_one_path(
      table_ids, path, std::to_string(mode));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "save model failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::SaveMultiTableOnePath does nothing when no pslib";
#endif
}

void FleetWrapper::SaveModel(const std::string& path, const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->save(path, std::to_string(mode));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "save model failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::SaveModel does nothing when no pslib";
#endif
}

void FleetWrapper::SaveModelOneTable(const uint64_t table_id,
                                     const std::string& path,
                                     const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret =
      pslib_ptr_->_worker_ptr->save(table_id, path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "save model of table id: " << table_id
               << ", to path: " << path << " failed";
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::SaveModelOneTable does nothing when no pslib";
#endif
}

void FleetWrapper::SaveModelOneTablePrefix(const uint64_t table_id,
                                           const std::string& path,
                                           const int mode,
                                           const std::string& prefix) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->save(
      table_id, path, std::to_string(mode), prefix);
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "save model (with prefix) of table id: " << table_id
               << ", to path: " << path << " failed";
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::SaveModelOneTablePrefix does nothing when no pslib";
#endif
}

void FleetWrapper::SetDate(const uint64_t table_id, const std::string& date) {
#if (defined PADDLE_WITH_PSLIB) && (defined PADDLE_WITH_HETERPS)
  assert(date.size() == 8);
  int year = std::stoi(date.substr(0, 4));
  int month = std::stoi(date.substr(4, 2));
  int day = std::stoi(date.substr(6, 2));
  struct std::tm b;
  b.tm_year = year - 1900;
  b.tm_mon = month - 1;
  b.tm_mday = day;
  b.tm_hour = b.tm_min = b.tm_sec = 0;
  std::time_t seconds_from_1970 = std::mktime(&b);
  int day_id = seconds_from_1970 / 86400;
  auto ret = pslib_ptr_->_worker_ptr->set_day_id(table_id, day_id);
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "setdate : " << date << " failed";
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::SetDate does nothing when no pslib-gpu";
#endif
}

void FleetWrapper::PrintTableStat(uint64_t table_id,
                                  uint32_t pass_id,
                                  size_t threshold) {
#ifdef PADDLE_WITH_PSLIB
  auto ret =
      pslib_ptr_->_worker_ptr->print_table_stat(table_id, pass_id, threshold);
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "print table stat failed";
  }
#else
  VLOG(0) << "FleetWrapper::PrintTableStat does nothing when no pslib";
#endif
}

void FleetWrapper::SetFileNumOneShard(const uint64_t table_id, int file_num) {
#if (defined PADDLE_WITH_PSLIB) && (defined PADDLE_WITH_HETERPS)
  auto ret =
      pslib_ptr_->_worker_ptr->set_file_num_one_shard(table_id, file_num);
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "set_file_num_one_shard failed";
  }
#else
  VLOG(0) << "FleetWrapper::SetFileNumOneShard does nothing when no pslib-gpu";
#endif
}

double FleetWrapper::GetCacheThreshold(int table_id) {
#ifdef PADDLE_WITH_PSLIB
  double cache_threshold = 0.0;
  auto ret = pslib_ptr_->_worker_ptr->flush();
  ret.wait();
  ret = pslib_ptr_->_worker_ptr->get_cache_threshold(table_id, cache_threshold);
  ret.wait();
  if (cache_threshold < 0) {
    LOG(ERROR) << "get cache threshold failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return cache_threshold;
#else
  VLOG(0) << "FleetWrapper::GetCacheThreshold does nothing when no pslib";
  return 0.0;
#endif
}

void FleetWrapper::CacheShuffle(int table_id,
                                const std::string& path,
                                const int mode,
                                const double cache_threshold) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->cache_shuffle(
      table_id, path, std::to_string(mode), std::to_string(cache_threshold));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "cache shuffle failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::CacheShuffle does nothing when no pslib";
#endif
}

int32_t FleetWrapper::SaveCache(int table_id,
                                const std::string& path,
                                const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret =
      pslib_ptr_->_worker_ptr->save_cache(table_id, path, std::to_string(mode));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "table save cache failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return feasign_cnt;
#else
  VLOG(0) << "FleetWrapper::SaveCache does nothing when no pslib";
  return -1;
#endif
}

int32_t FleetWrapper::SaveWithWhitelist(int table_id,
                                        const std::string& path,
                                        const int mode,
                                        const std::string& whitelist_path) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->save_with_whitelist(
      table_id, path, std::to_string(mode), whitelist_path);
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "table save cache failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return feasign_cnt;
#else
  VLOG(0) << "FleetWrapper::SaveCache does nothing when no pslib";
  return -1;
#endif
}

void FleetWrapper::ShrinkSparseTable(int table_id) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->shrink(table_id);
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "Shrink Sparse Table failed";
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::ShrinkSparseTable does nothing when no pslib";
#endif
}

void FleetWrapper::ClearModel() {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->clear();
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "Clear Model failed";
  }
#else
  VLOG(0) << "FleetWrapper::ClearModel does nothing when no pslib";
#endif
}

void FleetWrapper::ClearOneTable(const uint64_t table_id) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->clear(table_id);
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "Clear One Table failed table_id: " << table_id;
  }
#else
  VLOG(0) << "FleetWrapper::ClearOneTable does nothing when no pslib";
#endif
}

void FleetWrapper::ShrinkDenseTable(int table_id,
                                    Scope* scope,
                                    std::vector<std::string> var_list,
                                    float decay,
                                    int emb_dim) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<paddle::ps::Region> regions;
  for (std::string& name : var_list) {
    if (name.find("batch_sum") != std::string::npos) {
      Variable* var = scope->FindVar(name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          common::errors::NotFound("Variable (var) is null. var[%s] not found.",
                                   name));

      VLOG(0) << "prepare shrink dense batch_sum";
      phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
      float* g = tensor->data<float>();

      // show_batch_sum += N * log(decay)
      std::string size_name = name;
      size_name.replace(
          size_name.find("batch_sum"), size_name.length(), "batch_size");
      Variable* var_size = scope->FindVar(size_name);
      PADDLE_ENFORCE_NOT_NULL(
          var_size,
          common::errors::NotFound("Variable size is null. var[%s] not found.",
                                   size_name));

      VLOG(3) << "shrink dense batch_sum: " << name << ", " << size_name;
      float* g_size = var_size->GetMutable<phi::DenseTensor>()->data<float>();

      for (int k = 0; k < tensor->numel(); k += emb_dim) {
        g[k] = g[k] + g_size[k] * log(decay);
      }
      paddle::ps::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    } else {
      Variable* var = scope->FindVar(name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          common::errors::NotFound("Variable (var) is null. var[%s] not found.",
                                   name));

      phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
      float* g = tensor->data<float>();
      paddle::ps::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    }
  }
  auto push_status = pslib_ptr_->_worker_ptr->push_dense_param(
      regions.data(), regions.size(), table_id);
  push_status.wait();
  auto status = push_status.get();
  if (status != 0) {
    // PADDLE_THROW(common::errors::Fatal(
    //    "push shrink dense param failed, status is [%d].", status));
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::ShrinkSparseTable does nothing when no pslib";
#endif
}

void FleetWrapper::ClientFlush() {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->flush();
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "Client Flush failed";
  }
#else
  VLOG(0) << "FleetWrapper::ServerFlush does nothing when no pslib";
#endif
}

int FleetWrapper::RegisterClientToClientMsgHandler(int msg_type,
                                                   MsgHandlerFunc handler) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "calling FleetWrapper::RegisterClientToClientMsgHandler";
  VLOG(3) << "pslib_ptr_=" << pslib_ptr_;
  VLOG(3) << "_worker_ptr=" << pslib_ptr_->_worker_ptr;
  return pslib_ptr_->_worker_ptr->register_client2client_msg_handler(msg_type,
                                                                     handler);
#else
  VLOG(0) << "FleetWrapper::RegisterClientToClientMsgHandler"
          << " does nothing when no pslib";
#endif
  return 0;
}

std::future<int32_t> FleetWrapper::SendClientToClientMsg(
    int msg_type, int to_client_id, const std::string& msg) {
#ifdef PADDLE_WITH_PSLIB
  return pslib_ptr_->_worker_ptr->send_client2client_msg(
      msg_type, to_client_id, msg);
#else
  VLOG(0) << "FleetWrapper::SendClientToClientMsg"
          << " does nothing when no pslib";
#endif
  return std::future<int32_t>();
}

std::default_random_engine& FleetWrapper::LocalRandomEngine() {
  struct engine_wrapper_t {
    std::default_random_engine engine;
#ifdef PADDLE_WITH_PSLIB
    engine_wrapper_t() {
      struct timespec tp;
      clock_gettime(CLOCK_REALTIME, &tp);
      double cur_time = tp.tv_sec + tp.tv_nsec * 1e-9;
      static std::atomic<uint64_t> x(0);
      std::seed_seq sseq = {x++, x++, x++, (uint64_t)(cur_time * 1000)};
      engine.seed(sseq);
    }
#endif
  };
  thread_local engine_wrapper_t r;
  return r.engine;
}

int32_t FleetWrapper::CopyTable(const uint64_t src_table_id,
                                const uint64_t dest_table_id) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->copy_table(src_table_id, dest_table_id);
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "copy table failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return feasign_cnt;
#else
  VLOG(0) << "FleetWrapper::CopyTable does nothing when no pslib";
  return 0;
#endif
}

void FleetWrapper::Confirm() {
#ifdef PADDLE_WITH_PSLIB
  // FIXME(xujiaqi01): will later support confirm
  // auto ret = pslib_ptr_->_worker_ptr->confirm();
  // ret.wait();
  VLOG(0) << "disable FleetWrapper::Confirm temporarily";
#else
  VLOG(0) << "FleetWrapper::Confirm does nothing when no pslib";
#endif
}

void FleetWrapper::Revert() {
#ifdef PADDLE_WITH_PSLIB
  // FIXME(xujiaqi01): will later support revert
  // auto ret = pslib_ptr_->_worker_ptr->revert();
  // ret.wait();
  VLOG(0) << "disable FleetWrapper::Revert temporarily";
#else
  VLOG(0) << "FleetWrapper::Revert does nothing when no pslib";
#endif
}

int32_t FleetWrapper::CopyTableByFeasign(
    const uint64_t src_table_id,
    const uint64_t dest_table_id,
    const std::vector<uint64_t>& feasign_list) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->copy_table_by_feasign(
      src_table_id, dest_table_id, feasign_list.data(), feasign_list.size());
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "copy table by feasign failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return feasign_cnt;
#else
  VLOG(0) << "FleetWrapper::CopyTableByFeasign does nothing when no pslib";
  return 0;
#endif
}

size_t FleetWrapper::GetAbsoluteSum(size_t start,
                                    size_t end,
                                    size_t level,
                                    const phi::LegacyLoD& lod) {
  if (level >= lod.size() - 1) {
    return end - start;
  }
  size_t ret = 0;
  for (size_t i = start; i < end - 1; ++i) {
    size_t pos1 = lod[level][i];
    size_t pos2 = lod[level][i + 1];
    ret += GetAbsoluteSum(pos1, pos2, level + 1, lod);
  }
  return ret;
}

}  // namespace paddle::framework
