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

#include <future>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/common/cost_timer.h"
#include "paddle/fluid/distributed/ps/service/env.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/ps/service/sparse_shard_value.h"
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"
#include "paddle/phi/core/platform/timer.h"

namespace paddle {
namespace distributed {

using paddle::distributed::PsRequestMessage;
using paddle::distributed::PsResponseMessage;

typedef std::function<void(void *)> PSClientCallBack;
class PSClientClosure : public google::protobuf::Closure {
 public:
  explicit PSClientClosure(PSClientCallBack callback) : _callback(callback) {}
  virtual ~PSClientClosure() {}
  virtual void set_promise_value(int value) {
    for (auto &promise : _promises) {
      promise->set_value(value);
    }
  }

  void add_promise(std::shared_ptr<std::promise<int32_t>> &promise) {  // NOLINT
    _promises.push_back(promise);
  }

  void add_timer(std::shared_ptr<CostTimer> &timer) {  // NOLINT
    _timers.push_back(timer);
  }

 protected:
  PSClientCallBack _callback;
  std::vector<std::shared_ptr<CostTimer>> _timers;
  std::vector<std::shared_ptr<std::promise<int32_t>>> _promises;
};

class PSClient {
 public:
  PSClient() {}
  virtual ~PSClient() {}
  PSClient(PSClient &&) = delete;
  PSClient(const PSClient &) = delete;

  virtual int32_t Configure(  // NOLINT
      const PSParameter &config,
      const std::map<uint64_t, std::vector<paddle::distributed::Region>>
          &regions,
      PSEnvironment &_env,  // NOLINT
      size_t client_id);

  virtual int32_t CreateClient2ClientConnection(int pserver_timeout_ms,
                                                int pserver_connect_timeout_ms,
                                                int max_retry) = 0;

  // 触发table数据退场
  virtual std::future<int32_t> Shrink(uint32_t table_id,
                                      const std::string threshold) = 0;

  // 全量table进行数据load
  virtual std::future<int32_t> Load(const std::string &epoch,
                                    const std::string &mode) = 0;
  // 指定table数据load
  virtual std::future<int32_t> Load(uint32_t table_id,
                                    const std::string &epoch,
                                    const std::string &mode) = 0;

  // 全量table数据save  value_accessor根据mode，可能有不同的save条件
  virtual std::future<int32_t> Save(const std::string &epoch,
                                    const std::string &mode) = 0;
  // 指定table数据save  value_accessor根据mode，可能有不同的save条件
  virtual std::future<int32_t> Save(uint32_t table_id,
                                    const std::string &epoch,
                                    const std::string &mode) = 0;

  // 清空table数据
  virtual std::future<int32_t> Clear() = 0;
  virtual std::future<int32_t> Clear(uint32_t table_id) = 0;

  // pull dense的参数部分，并分块填充到本地网络参数中
  // start和num用于拉取部分参数
  // future结束前keys和values缓冲区不能再次使用
  // client将values按照区块拆包后送交多个sender
  // sender聚集同一区块的请求，累计多个填充buffer
  // server将参数区块中配置的某一维提取返回
  // 返回数据解包后填充到累计的多个buffer中
  virtual std::future<int32_t> PullDense(Region *regions,
                                         size_t region_num,
                                         size_t table_id) = 0;  // 保留

  // firstly push dense param for parameter server
  // this is necessary because dense weight initialized in trainer on cold
  // start
  virtual std::future<int32_t> PushDenseParam(const Region *regions,
                                              size_t region_num,
                                              size_t table_id) = 0;

  virtual std::future<int32_t> PushDense(const Region *regions,
                                         size_t region_num,
                                         size_t table_id) = 0;

  // 使用keys进行pull请求，结果填充values
  // keys和values的个数均为num个，每个value占用select_size空间
  // future结束前keys和values缓冲区不能再次使用
  // 整合多个线程请求的keys，聚集并分散发送到server
  // 返回结果后，遍历buffer并对values赋值
  // is_training 用于区分请求是训练/预测，server端对于特征和准入会有不同的处理.
  virtual std::future<int32_t> PullSparse(float **select_values,
                                          size_t table_id,
                                          const uint64_t *keys,
                                          size_t num,
                                          bool is_training) = 0;

  virtual std::future<int32_t> PullSparseParam(float **select_values UNUSED,
                                               size_t table_id UNUSED,
                                               const uint64_t *keys UNUSED,
                                               size_t num UNUSED,
                                               bool is_training UNUSED) {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  virtual ::std::future<int32_t> PullSparsePtr(
      int shard_id UNUSED,
      char **select_values UNUSED,
      size_t table_id UNUSED,
      const uint64_t *keys UNUSED,
      size_t num UNUSED,
      uint16_t pass_id UNUSED,
      const std::vector<std::unordered_map<uint64_t, uint32_t>> &keys2rank_vec
          UNUSED,
      const uint16_t &dim_id UNUSED = 0) {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }
  virtual ::std::future<int32_t> PullSparseKey(
      int shard_id UNUSED,
      size_t table_id UNUSED,
      const uint64_t *keys UNUSED,
      size_t num UNUSED,
      uint16_t pass_id UNUSED,
      const std::vector<std::unordered_map<uint64_t, uint32_t>> &keys2rank_vec
          UNUSED,
      const uint16_t &dim_id UNUSED = 0) {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  virtual std::future<int32_t> PrintTableStat(uint32_t table_id,
                                              uint16_t pass_id,
                                              size_t threshold) = 0;
  virtual std::future<int32_t> SaveCacheTable(uint32_t table_id UNUSED,
                                              uint16_t pass_id UNUSED,
                                              size_t threshold UNUSED) {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  // 确保所有积攒中的请求都发起发送
  virtual std::future<int32_t> Flush() = 0;
  // server优雅退出
  virtual std::future<int32_t> StopServer() = 0;

  // server profiler
  virtual std::future<int32_t> StartProfiler() = 0;
  virtual std::future<int32_t> StopProfiler() = 0;

  virtual std::future<int32_t> Barrier(size_t table_id,
                                       uint32_t barrier_type) = 0;

  virtual std::future<int32_t> PullGeoParam(size_t table_id,
                                            std::vector<float> *values,
                                            std::vector<uint64_t> *keys,
                                            int pserver_idx) = 0;

  virtual std::future<int32_t> PushGlobalStep(int table_id,
                                              int64_t *total_send_data,
                                              void *done) = 0;

  // recv table from server and save it in DenseTensor
  virtual int32_t RecvAndSaveTable(const uint64_t table_id,
                                   const std::string &path) = 0;

  virtual void FinalizeWorker() = 0;
  // client to client, 消息发送
  virtual std::future<int32_t> SendClient2ClientMsg(int msg_type UNUSED,
                                                    int to_client_id UNUSED,
                                                    const std::string &msg
                                                        UNUSED) {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  // client2client消息处理，std::function<int32_t (int, int, const std::string&)
  // -> ret (msg_type, from_client_id, msg)
  typedef std::function<int32_t(int, int, const std::string &)> MsgHandlerFunc;
  virtual int RegisterClient2ClientMsgHandler(int msg_type,
                                              MsgHandlerFunc handler) {
    _msg_handler_map[msg_type] = handler;
    return 0;
  }

  virtual int HandleClient2ClientMsg(int msg_type,
                                     int from_client_id,
                                     const std::string &msg) {
    auto itr = _msg_handler_map.find(msg_type);
    if (itr == _msg_handler_map.end()) {
      LOG(WARNING) << "unknown client2client_msg type:" << msg_type;
      return -1;
    }
    return itr->second(msg_type, from_client_id, msg);
  }

  virtual ValueAccessor *GetTableAccessor(size_t table_id) {
    auto itr = _table_accessors.find(table_id);
    if (itr == _table_accessors.end()) {
      return NULL;
    }
    return itr->second.get();
  }

  virtual size_t GetServerNums() = 0;

  virtual std::future<int32_t> PushDenseRawGradient(int table_id,
                                                    float *total_send_data,
                                                    size_t total_send_data_size,
                                                    void *done) = 0;

  virtual std::future<int32_t> PushSparseRawGradient(
      size_t table_id,
      const uint64_t *keys,
      const float **update_values,
      size_t num,
      void *done) = 0;

  virtual std::future<int32_t> PushSparseRawGradientPartial(
      size_t table_id,
      const uint64_t *keys,
      const float **update_values,
      uint32_t num,
      void *done,
      int pserver_idx) = 0;

  virtual std::future<int32_t> PushSparseParam(size_t table_id,
                                               const uint64_t *keys,
                                               const float **update_values,
                                               size_t num,
                                               void *done) = 0;
  virtual std::future<int32_t> PushSparse(size_t table_id,
                                          const uint64_t *keys,
                                          const float **update_values,
                                          size_t num) = 0;

  // for save cache
  virtual std::future<int32_t> CacheShuffle(uint32_t table_id UNUSED,
                                            const std::string &path UNUSED,
                                            const std::string &mode UNUSED,
                                            const std::string &cache_threshold
                                                UNUSED) {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  virtual std::future<int32_t> CacheShuffleMultiTable(
      std::vector<int> tables UNUSED,
      const std::string &path UNUSED,
      const std::string &mode UNUSED,
      const std::string &cache_threshold UNUSED) {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  virtual std::future<int32_t> SaveCache(uint32_t table_id UNUSED,
                                         const std::string &path UNUSED,
                                         const std::string &mode UNUSED) {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  virtual std::future<int32_t> GetCacheThreshold(uint32_t table_id UNUSED,
                                                 double &cache_threshold
                                                     UNUSED) {  // NOLINT
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  virtual std::future<int32_t> Revert() {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  virtual std::future<int32_t> CheckSavePrePatchDone() {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }
  // add
  virtual std::shared_ptr<SparseShardValues> TakePassSparseReferredValues(
      const size_t &table_id UNUSED,
      const uint16_t &pass_id UNUSED,
      const uint16_t &dim_id UNUSED) {
    VLOG(0) << "Did not implement";
    return nullptr;
  }

  virtual ::std::future<int32_t> SetDayId(size_t table_id, int day_id) {
    VLOG(0) << "SetDayId Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

 protected:
  virtual int32_t Initialize() = 0;
  PSParameter _config;
  std::map<uint64_t, std::vector<paddle::distributed::Region>>
      _dense_pull_regions;
  std::unordered_map<uint32_t, std::shared_ptr<ValueAccessor>> _table_accessors;
  std::unordered_map<int32_t, MsgHandlerFunc>
      _msg_handler_map;  // 处理client2client消息

 public:
  size_t _client_id;
  PSEnvironment *_env;
};

template <class T>
class AsyncRequestTask {
 public:
  AsyncRequestTask() : _promise(std::make_shared<std::promise<int32_t>>()) {}
  AsyncRequestTask(T &data, size_t table_id, std::shared_ptr<CostTimer> &timer)
      : _table_id(table_id),
        _timer(timer),
        _promise(std::make_shared<std::promise<int32_t>>()) {
    _data = std::move(data);
  }

  AsyncRequestTask(AsyncRequestTask &data)  // NOLINT
      : _table_id(data.table_id()),
        _timer(data.timer()),
        _promise(data.promise()) {
    _data = std::move(data.data());
  }

  ~AsyncRequestTask() {}

  inline T &data() { return _data; }
  inline size_t table_id() { return _table_id; }
  inline std::shared_ptr<CostTimer> &timer() { return _timer; }
  inline std::future<int32_t> get_future() { return _promise->get_future(); }
  inline std::shared_ptr<std::promise<int32_t>> &promise() { return _promise; }

 private:
  T _data;
  size_t _table_id;
  std::shared_ptr<CostTimer> _timer;
  std::shared_ptr<std::promise<int32_t>> _promise;
};

REGISTER_PSCORE_REGISTERER(PSClient);

class PSClientFactory {
 public:
  static PSClient *Create(const PSParameter &config);
};
}  // namespace distributed
}  // namespace paddle
