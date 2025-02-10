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
#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/distributed/ps/service/ps_graph_client.h"
#include "paddle/fluid/distributed/ps/service/simple_rpc/rpc_server.h"
#include "paddle/fluid/distributed/ps/table/table.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
// #include "paddle/fluid/framework/threadpool.h"

namespace paddle {
namespace distributed {
PsGraphClient::PsGraphClient() {
  simple::global_rpc_server().initialize();
  auto gloo = paddle::framework::GlooWrapper::GetInstance();
  _rank_id = gloo->Rank();
  _rank_num = gloo->Size();
  _service = simple::global_rpc_server().add_service(
      [this](const simple::RpcMessageHead &head,
             paddle::framework::BinaryArchive &iar) {
        request_handler(head, iar);
      });
  _partition_key_service = simple::global_rpc_server().add_service(
      [this](const simple::RpcMessageHead &head,
             paddle::framework::BinaryArchive &iar) {
        request_key_handler(head, iar);
      });
  VLOG(0) << "PsGraphClient rank id=" << _rank_id << ", rank num=" << _rank_num;
}
PsGraphClient::~PsGraphClient() {}
int32_t PsGraphClient::Initialize() {
  const auto &downpour_param = _config.server_param().downpour_server_param();
  uint32_t max_shard_num = 0;
  for (int i = 0; i < downpour_param.downpour_table_param_size(); ++i) {
    auto &param = downpour_param.downpour_table_param(i);
    uint32_t table_id = param.table_id();
    uint32_t shard_num = param.shard_num();
    _table_info[table_id] = std::make_shared<SparseTableInfo>();
    _table_info[table_id]->shard_num = shard_num;
    if (max_shard_num < shard_num) {
      max_shard_num = shard_num;
    }
  }
  for (uint32_t k = 0; k < max_shard_num; ++k) {
    _thread_pools.push_back(std::make_shared<phi::ThreadPool>(1));
  }
  _local_shard_keys.resize(max_shard_num);
  _shard_ars.resize(max_shard_num);

  return PsLocalClient::Initialize();
}
void PsGraphClient::FinalizeWorker() {
  if (_service != nullptr) {
    simple::global_rpc_server().remove_service(_service);
    _service = nullptr;
    fprintf(stdout, "FinalizeWorker remove rpc service");
  }
  if (_partition_key_service != nullptr) {
    simple::global_rpc_server().remove_service(_partition_key_service);
    _partition_key_service = nullptr;
    fprintf(stdout, "FinalizeWorker remove rpc partition_key_service");
  }
  simple::global_rpc_server().finalize();
}
// add macro
#define DIM_PASS_ID(dim_id, pass_id) \
  uint32_t((uint32_t(dim_id) << 16) | pass_id)
#define GET_PASS_ID(id) (id & 0xffff)
#define GET_DIM_ID(id) ((id >> 16) & 0xffff)

::std::future<int32_t> PsGraphClient::PullSparsePtr(
    int shard_id,
    char **select_values,
    size_t table_id,
    const uint64_t *keys,
    size_t num,
    uint16_t pass_id,
    const std::vector<std::unordered_map<uint64_t, uint32_t>> &keys2rank_vec,
    const uint16_t &dim_id) {
  platform::Timer timeline;
  timeline.Start();
  // ps_gpu_wrapper
  auto ps_wrapper = paddle::framework::PSGPUWrapper::GetInstance();

  std::vector<uint64_t> &local_keys = _local_shard_keys[shard_id];
  local_keys.clear();

  auto &ars = _shard_ars[shard_id];
  ars.resize(_rank_num);
  for (int rank = 0; rank < _rank_num; ++rank) {
    ars[rank].Clear();
  }

  // split keys to rankid
  for (size_t i = 0; i < num; ++i) {
    auto &k = keys[i];
    int rank = 0;
    auto shard_num = keys2rank_vec.size();
    if (shard_num > 0) {
      auto shard = k % shard_num;
      auto it = keys2rank_vec[shard].find(k);
      if (it != keys2rank_vec[shard].end()) {
        rank = it->second;
      } else {
        // Should not happen
        VLOG(0) << "PullSparsePtr, miss key " << k << " rank=" << _rank_id;
        PADDLE_ENFORCE_NE(it,
                          keys2rank_vec[shard].end(),
                          common::errors::InvalidArgument(
                              "The key was not found in the expected shard."));
      }
    } else {
      rank = ps_wrapper->PartitionKeyForRank(k);
    }
    if (rank == _rank_id) {
      local_keys.push_back(k);
    } else {
      ars[rank].PutRaw(k);
    }
  }
  paddle::framework::WaitGroup wg;
  wg.add(_rank_num);

  uint32_t id = DIM_PASS_ID(dim_id, pass_id);
  // send to remote
  for (int rank = 0; rank < _rank_num; ++rank) {
    if (rank == _rank_id) {
      wg.done();
      continue;
    }
    auto &ar = ars[rank];
    size_t n = ar.Length() / sizeof(uint64_t);
    ar.PutRaw(n);
    ar.PutRaw(shard_id);
    ar.PutRaw(id);
    simple::global_rpc_server().send_request_consumer(
        rank,
        table_id,
        _service,
        ar,
        [this, &wg](const simple::RpcMessageHead & /**head*/,
                    framework::BinaryArchive & /**ar*/) { wg.done(); });
  }
  // not empty
  if (!local_keys.empty()) {
    auto f = _thread_pools[shard_id]->Run(
        [this, table_id, pass_id, shard_id, &local_keys, &select_values](void) {
          // local pull values
          Table *table_ptr = GetTable(table_id);
          TableContext table_context;
          table_context.value_type = Sparse;
          table_context.pull_context.keys = &local_keys[0];
          table_context.pull_context.ptr_values = select_values;
          table_context.use_ptr = true;
          table_context.num = local_keys.size();
          table_context.shard_id = shard_id;
          table_context.pass_id = pass_id;
          table_ptr->Pull(table_context);
        });
    f.get();
  }
  wg.wait();
  timeline.Pause();
  VLOG(3) << "PullSparsePtr local table id=" << table_id
          << ", pass id=" << pass_id << ", shard_id=" << shard_id
          << ", dim_id=" << dim_id << ", keys count=" << num
          << ", span=" << timeline.ElapsedSec();

  return done();
}

::std::future<int32_t> PsGraphClient::PullSparseKey(
    int shard_id,
    size_t table_id,
    const uint64_t *keys,
    size_t num,
    uint16_t pass_id,
    const std::vector<std::unordered_map<uint64_t, uint32_t>> &keys2rank_vec,
    const uint16_t &dim_id) {
  platform::Timer timeline;
  timeline.Start();
  // ps_gpu_wrapper
  auto ps_wrapper = paddle::framework::PSGPUWrapper::GetInstance();

  std::vector<uint64_t> &local_keys = _local_shard_keys[shard_id];
  local_keys.clear();

  auto &ars = _shard_ars[shard_id];
  ars.resize(_rank_num);
  for (int rank = 0; rank < _rank_num; ++rank) {
    ars[rank].Clear();
  }
  // split keys to rankid
  for (size_t i = 0; i < num; ++i) {
    auto &k = keys[i];
    int rank = 0;
    auto shard_num = keys2rank_vec.size();
    if (shard_num > 0) {
      auto shard = k % shard_num;
      auto it = keys2rank_vec[shard].find(k);
      if (it != keys2rank_vec[shard].end()) {
        rank = it->second;
      } else {
        VLOG(0) << "PullSparseKey, miss key " << k << " rank=" << _rank_id;
        PADDLE_ENFORCE_NE(
            it,
            keys2rank_vec[shard].end(),
            common::errors::InvalidArgument(
                "The key was not found in the expected shard.", k));
      }
    } else {
      rank = ps_wrapper->PartitionKeyForRank(k);
    }
    if (rank == _rank_id) {
      local_keys.push_back(k);
    } else {
      ars[rank].PutRaw(k);
    }
  }
  paddle::framework::WaitGroup wg;
  wg.add(_rank_num);

  uint32_t id = DIM_PASS_ID(dim_id, pass_id);
  // send to remote
  for (int rank = 0; rank < _rank_num; ++rank) {
    if (rank == _rank_id) {
      wg.done();
      continue;
    }
    auto &ar = ars[rank];
    size_t n = ar.Length() / sizeof(uint64_t);
    ar.PutRaw(n);
    ar.PutRaw(shard_id);
    ar.PutRaw(id);
    simple::global_rpc_server().send_request_consumer(
        rank,
        table_id,
        _partition_key_service,
        ar,
        [this, &wg](const simple::RpcMessageHead & /**head*/,
                    framework::BinaryArchive & /**ar*/) { wg.done(); });
  }

  wg.wait();
  timeline.Pause();
  VLOG(3) << "PullSparseKey local table id=" << table_id
          << ", pass id=" << pass_id << ", shard_id=" << shard_id
          << ", dim_id=" << dim_id << ", keys count=" << num
          << ", span=" << timeline.ElapsedSec();

  return done();
}

// server pull remote keys values
void PsGraphClient::request_handler(const simple::RpcMessageHead &head,
                                    paddle::framework::BinaryArchive &iar) {
  size_t table_id = head.consumer_id;
  uint32_t id = 0;
  iar.ReadBack(&id, sizeof(uint32_t));
  int shard_id = 0;
  iar.ReadBack(&shard_id, sizeof(int));
  size_t num = 0;
  iar.ReadBack(&num, sizeof(size_t));

  SparsePassValues *pass_referred = nullptr;
  SparseTableInfo &info = get_table_info(table_id);
  info.pass_mutex.lock();
  auto it = info.referred_feas.find(id);
  if (it == info.referred_feas.end()) {
    pass_referred = new SparsePassValues;
    pass_referred->wg.clear();
    int total_ref = info.shard_num * (_rank_num - 1);
    pass_referred->wg.add(total_ref);
    pass_referred->values = new SparseShardValues;
    pass_referred->shard_mutex = new std::mutex[info.shard_num];
    pass_referred->values->resize(info.shard_num);
    info.referred_feas[id].reset(pass_referred);
    info.sem_wait.post();
    VLOG(0) << "add request_handler table id=" << table_id
            << ", pass id=" << GET_PASS_ID(id) << ", shard_id=" << shard_id
            << ", total_ref=" << total_ref;
  } else {
    pass_referred = it->second.get();
  }
  info.pass_mutex.unlock();

  auto &shard_values = (*pass_referred->values)[shard_id];
  auto &shard_mutex = pass_referred->shard_mutex[shard_id];
  shard_mutex.lock();
  size_t shard_size = shard_values.keys.size();
  shard_values.offsets.push_back(shard_size);
  if (num > 0) {
    shard_values.keys.resize(num + shard_size);
    iar.Read(&shard_values.keys[shard_size], num * sizeof(uint64_t));
    shard_values.values.resize(num + shard_size);
  }
  shard_mutex.unlock();

  if (num > 0) {
    auto f = _thread_pools[shard_id]->Run(
        [this, table_id, id, shard_id, num, shard_size, pass_referred](void) {
          thread_local std::vector<uint64_t> pull_keys;
          thread_local std::vector<char *> pull_vals;

          pull_keys.resize(num);
          pull_vals.resize(num);

          auto &shard_values = (*pass_referred->values)[shard_id];
          auto &shard_mutex = pass_referred->shard_mutex[shard_id];
          shard_mutex.lock();
          uint64_t *keys = &shard_values.keys[shard_size];
          for (size_t i = 0; i < num; ++i) {
            pull_keys[i] = keys[i];
          }
          shard_mutex.unlock();

          platform::Timer timeline;
          timeline.Start();
          auto *table_ptr = GetTable(table_id);
          TableContext table_context;
          table_context.value_type = Sparse;
          table_context.pull_context.keys = &pull_keys[0];
          table_context.pull_context.ptr_values = &pull_vals[0];
          table_context.use_ptr = true;
          table_context.num = num;
          table_context.shard_id = shard_id;
          table_context.pass_id = GET_PASS_ID(id);
          table_ptr->Pull(table_context);
          timeline.Pause();

          shard_mutex.lock();
          char **valsptr = &shard_values.values[shard_size];
          for (size_t i = 0; i < num; ++i) {
            valsptr[i] = pull_vals[i];
          }
          shard_mutex.unlock();

          VLOG(3) << "end pull remote table id=" << table_id
                  << ", pass id=" << GET_PASS_ID(id)
                  << ", shard_id=" << shard_id << ", keys count=" << num
                  << ", span=" << timeline.ElapsedSec();
          // notify done
          pass_referred->wg.done();
        });
  } else {
    // zero done
    pass_referred->wg.done();
  }
  // send response
  paddle::framework::BinaryArchive oar;
  simple::global_rpc_server().send_response(head, oar);
}

// server pull remote keys (only key)
void PsGraphClient::request_key_handler(const simple::RpcMessageHead &head,
                                        paddle::framework::BinaryArchive &iar) {
  size_t table_id = head.consumer_id;
  uint32_t id = 0;
  iar.ReadBack(&id, sizeof(uint32_t));
  int shard_id = 0;
  iar.ReadBack(&shard_id, sizeof(int));
  size_t num = 0;
  iar.ReadBack(&num, sizeof(size_t));

  SparsePassValues *pass_referred = nullptr;
  SparseTableInfo &info = get_table_info(table_id);
  info.pass_mutex.lock();
  auto it = info.referred_feas.find(id);
  if (it == info.referred_feas.end()) {
    pass_referred = new SparsePassValues;
    pass_referred->wg.clear();
    int total_ref = info.shard_num * (_rank_num - 1);
    pass_referred->wg.add(total_ref);
    pass_referred->values = new SparseShardValues;
    pass_referred->shard_mutex = new std::mutex[info.shard_num];
    pass_referred->values->resize(info.shard_num);
    info.referred_feas[id].reset(pass_referred);
    info.sem_wait.post();
    VLOG(0) << "add request_handler table id=" << table_id
            << ", pass id=" << GET_PASS_ID(id) << ", shard_id=" << shard_id
            << ", total_ref=" << total_ref;
  } else {
    pass_referred = it->second.get();
  }
  info.pass_mutex.unlock();

  auto &shard_values = (*pass_referred->values)[shard_id];
  auto &shard_mutex = pass_referred->shard_mutex[shard_id];
  shard_mutex.lock();
  size_t shard_size = shard_values.keys.size();
  shard_values.offsets.push_back(shard_size);
  if (num > 0) {
    shard_values.keys.resize(num + shard_size);
    iar.Read(&shard_values.keys[shard_size], num * sizeof(uint64_t));
  }
  shard_mutex.unlock();
  pass_referred->wg.done();
  // send response
  paddle::framework::BinaryArchive oar;
  simple::global_rpc_server().send_response(head, oar);
}

// get shard num
PsGraphClient::SparseTableInfo &PsGraphClient::get_table_info(
    const size_t &table_id) {
  return (*_table_info[table_id].get());
}
// get pass keep keys values
std::shared_ptr<SparseShardValues> PsGraphClient::TakePassSparseReferredValues(
    const size_t &table_id, const uint16_t &pass_id, const uint16_t &dim_id) {
  SparseTableInfo &info = get_table_info(table_id);
  uint32_t id = DIM_PASS_ID(dim_id, pass_id);
  info.sem_wait.wait();
  SparsePassValues *pass_referred = nullptr;

  info.pass_mutex.lock();
  auto it = info.referred_feas.find(id);
  if (it == info.referred_feas.end()) {
    info.pass_mutex.unlock();
    VLOG(0) << "table_id=" << table_id
            << ", TakePassSparseReferredValues pass_id=" << pass_id
            << ", dim_id=" << dim_id << " is nullptr";
    return nullptr;
  }
  pass_referred = it->second.get();
  info.pass_mutex.unlock();

  int cnt = pass_referred->wg.count();
  VLOG(0) << "table_id=" << table_id
          << ", begin TakePassSparseReferredValues pass_id=" << pass_id
          << ", dim_id=" << dim_id << " wait count=" << cnt;
  pass_referred->wg.wait();

  std::shared_ptr<SparseShardValues> shard_ptr;
  shard_ptr.reset(pass_referred->values);
  pass_referred->values = nullptr;
  // free shard mutex lock
  delete[] pass_referred->shard_mutex;

  info.pass_mutex.lock();
  info.referred_feas.erase(id);
  info.pass_mutex.unlock();

  return shard_ptr;
}
}  // namespace distributed
}  // namespace paddle
#endif
