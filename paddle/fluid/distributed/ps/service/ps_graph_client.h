// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License 0//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
#pragma once
#include "paddle/fluid/distributed/ps/service/ps_local_client.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/framework/barrier.h"
#include "paddle/fluid/framework/threadpool.h"

namespace paddle {
// namespace framework {
// class ThreadPool;
// };
namespace distributed {
namespace simple {
struct RpcMessageHead;
};

struct SparsePassValues {
  paddle::framework::WaitGroup wg;
  std::mutex *shard_mutex;
  SparseShardValues *values;
};
class PsGraphClient : public PsLocalClient {
  typedef std::unordered_map<uint32_t, std::shared_ptr<SparsePassValues>>
      SparseFeasReferredMap;
  struct SparseTableInfo {
    uint32_t shard_num;
    std::mutex pass_mutex;
    SparseFeasReferredMap referred_feas;
    paddle::framework::Semaphore sem_wait;
  };

 public:
  PsGraphClient();
  virtual ~PsGraphClient();
  virtual int32_t Initialize();
  virtual void FinalizeWorker();
  virtual ::std::future<int32_t> PullSparsePtr(
      int shard_id,
      char **select_values,
      size_t table_id,
      const uint64_t *keys,
      size_t num,
      uint16_t pass_id,
      const std::vector<std::unordered_map<uint64_t, uint32_t>> &keys2rank_vec,
      const uint16_t &dim_id = 0);

  virtual ::std::future<int32_t> PullSparseKey(
      int shard_id,
      size_t table_id,
      const uint64_t *keys,
      size_t num,
      uint16_t pass_id,
      const std::vector<std::unordered_map<uint64_t, uint32_t>> &keys2rank_vec,
      const uint16_t &dim_id = 0);

  virtual std::shared_ptr<SparseShardValues> TakePassSparseReferredValues(
      const size_t &table_id, const uint16_t &pass_id, const uint16_t &dim_id);

 public:
  void request_handler(const simple::RpcMessageHead &head,
                       paddle::framework::BinaryArchive &iar);  // NOLINT
  void request_key_handler(const simple::RpcMessageHead &head,
                           paddle::framework::BinaryArchive &iar);  // NOLINT
  SparseTableInfo &get_table_info(const size_t &table_id);

 private:
  std::map<uint32_t, std::shared_ptr<SparseTableInfo>> _table_info;
  void *_service = nullptr;
  void *_partition_key_service = nullptr;
  int _rank_id = 0;
  int _rank_num = 0;
  std::vector<std::shared_ptr<phi::ThreadPool>> _thread_pools;
  std::vector<std::vector<uint64_t>> _local_shard_keys;
  std::vector<std::vector<paddle::framework::BinaryArchive>> _shard_ars;
};
}  // namespace distributed
}  // namespace paddle
#endif
