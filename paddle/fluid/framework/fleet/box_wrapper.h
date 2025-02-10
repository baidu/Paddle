/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_BOX_PS
#include <afs_filesystem.h>
#include <boxps_public.h>
#include <dirent.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <ctime>
#include <deque>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/timer.h"
#include "paddle/utils/string/string_helper.h"
#define BUF_SIZE 1024 * 1024

extern void comlog_set_log_level(int log_level);
extern int com_logstatus();
namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_BOX_PS
class BasicAucCalculator {
 public:
  BasicAucCalculator() {}
  void init(int table_size) { set_table_size(table_size); }
  void reset() {
    for (int i = 0; i < 2; i++) {
      _table[i].assign(_table_size, 0.0);
    }
    _local_abserr = 0;
    _local_sqrerr = 0;
    _local_pred = 0;
  }
  void add_data(double pred, int label) {
    PADDLE_ENFORCE_GE(
        pred,
        0.0,
        common::errors::PreconditionNotMet("pred should be greater than 0"));
    PADDLE_ENFORCE_LE(
        pred,
        1.0,
        common::errors::PreconditionNotMet("pred should be lower than 1"));
    PADDLE_ENFORCE_EQ(
        label * label,
        label,
        common::errors::PreconditionNotMet(
            "label must be equal to 0 or 1, but its value is: %d", label));
    int pos = std::min(static_cast<int>(pred * _table_size), _table_size - 1);
    PADDLE_ENFORCE_GE(
        pos,
        0,
        common::errors::PreconditionNotMet(
            "pos must be equal or greater than 0, but its value is: %d", pos));
    PADDLE_ENFORCE_LT(
        pos,
        _table_size,
        common::errors::PreconditionNotMet(
            "pos must be less than table_size, but its value is: %d", pos));
    std::lock_guard<std::mutex> lock(_table_mutex);
    _local_abserr += fabs(pred - label);
    _local_sqrerr += (pred - label) * (pred - label);
    _local_pred += pred;
    _table[label][pos]++;
  }
  void compute();
  int table_size() const { return _table_size; }
  double bucket_error() const { return _bucket_error; }
  double auc() const { return _auc; }
  double mae() const { return _mae; }
  double actual_ctr() const { return _actual_ctr; }
  double predicted_ctr() const { return _predicted_ctr; }
  double size() const { return _size; }
  double rmse() const { return _rmse; }
  std::vector<double>& get_negative() { return _table[0]; }
  std::vector<double>& get_positive() { return _table[1]; }
  double& local_abserr() { return _local_abserr; }
  double& local_sqrerr() { return _local_sqrerr; }
  double& local_pred() { return _local_pred; }
  void calculate_bucket_error();

 protected:
  double _local_abserr = 0;
  double _local_sqrerr = 0;
  double _local_pred = 0;
  double _auc = 0;
  double _mae = 0;
  double _rmse = 0;
  double _actual_ctr = 0;
  double _predicted_ctr = 0;
  double _size;
  double _bucket_error = 0;

 private:
  void set_table_size(int table_size) {
    _table_size = table_size;
    for (int i = 0; i < 2; i++) {
      _table[i] = std::vector<double>();
    }
    reset();
  }
  int _table_size;
  std::vector<double> _table[2];
  static constexpr double kRelativeErrorBound = 0.05;
  static constexpr double kMaxSpan = 0.01;
  std::mutex _table_mutex;
};

class AfsStreamFile {
 public:
  explicit AfsStreamFile(afs::AfsFileSystem* afsfile)
      : afsfile_(afsfile), reader_(nullptr) {}
  virtual ~AfsStreamFile() {
    if (reader_ != NULL) {
      afsfile_->CloseReader(reader_);
      reader_ = NULL;
    }
  }
  virtual int Open(const char* path) {
    if (path == NULL) {
      return -1;
    }
    reader_ = afsfile_->OpenReader(path);
    PADDLE_ENFORCE_NE(reader_,
                      nullptr,
                      common::errors::PreconditionNotMet(
                          "OpenReader for file[%s] failed.", path));
    return 0;
  }
  virtual int Read(char* buf, int len) {
    int ret = reader_->Read(buf, len);
    return ret;
  }

 private:
  afs::AfsFileSystem* afsfile_;
  afs::Reader* reader_;
};

class AfsManager {
 public:
  AfsManager(const std::string& fs_name,
             const std::string& fs_ugi,
             const std::string& conf_path) {
    auto split = fs_ugi.find(",");
    std::string user = fs_ugi.substr(0, split);
    std::string pwd = fs_ugi.substr(split + 1);
    _afshandler = new afs::AfsFileSystem(
        fs_name.c_str(), user.c_str(), pwd.c_str(), conf_path.c_str());
    VLOG(0) << "AFSAPI Init: user: " << user << ", pwd: " << pwd;
    int ret = _afshandler->Init(true, (com_logstatus() == 0));
    PADDLE_ENFORCE_EQ(ret,
                      0,
                      common::errors::PreconditionNotMet(
                          "Called AFSAPI Init Interface Failed."));
    // Too high level will hurt the performance
    comlog_set_log_level(4);
    ret = _afshandler->Connect();
    PADDLE_ENFORCE_EQ(ret,
                      0,
                      common::errors::PreconditionNotMet(
                          "Called AFSAPI Connect Interface Failed"));
  }
  virtual ~AfsManager() {
    if (_afshandler != NULL) {
      _afshandler->DisConnect();
      _afshandler->Destroy();
      delete _afshandler;
      _afshandler = nullptr;
    }
  }
  static void ReadFromAfs(const std::string& path,
                          FILE* wfp,
                          afs::AfsFileSystem* _afshandler) {
    AfsStreamFile* read_stream = new AfsStreamFile(_afshandler);
    int ret = read_stream->Open(path.c_str());
    PADDLE_ENFORCE_EQ(ret,
                      0,
                      common::errors::PreconditionNotMet(
                          "Called AFSAPI Open file %s Failed.", path.c_str()));
    char* _buff = static_cast<char*>(calloc(BUF_SIZE + 2, sizeof(char)));
    int size = 0;
    while ((size = read_stream->Read(_buff, BUF_SIZE)) > 0) {
      fwrite(_buff, 1, size, wfp);
    }
    fflush(wfp);
    fclose(wfp);
    delete _buff;
    delete read_stream;
  }
  int PopenBidirectionalInternal(const char* command,
                                 FILE*& fp_read,   // NOLINT
                                 FILE*& fp_write,  // NOLINT
                                 pid_t& pid,       // NOLINT
                                 bool read,        // NOLINT
                                 bool write) {
    std::lock_guard<std::mutex> g(g_flock);
    int fd_read[2];
    int fd_write[2];
    if (read) {
      PADDLE_ENFORCE_EQ(
          pipe(fd_read),
          0,
          common::errors::External("Create read pipe failed in AfsManager."));
    }
    if (write) {
      PADDLE_ENFORCE_EQ(
          pipe(fd_write),
          0,
          common::errors::External("Create write pipe failed in AfsManager."));
    }
    pid = vfork();
    PADDLE_ENFORCE_GE(
        pid,
        0,
        common::errors::External(
            "Failed to create a child process via fork in AfsManager."));
    if (pid == 0) {
      if (read) {
        PADDLE_ENFORCE_NE(
            dup2(fd_read[1], STDOUT_FILENO),
            -1,
            common::errors::External(
                "Failed to duplicate file descriptor via dup2 in AfsManager."));
        close(fd_read[1]);
        close(fd_read[0]);
      }

      if (write) {
        PADDLE_ENFORCE_NE(
            dup2(fd_write[0], STDIN_FILENO),
            -1,
            common::errors::External(
                "Failed to duplicate file descriptor via dup2 in AfsManager."));
        close(fd_write[0]);
        close(fd_write[1]);
      }

      struct dirent* item;
      DIR* dir = opendir("/proc/self/fd");
      while ((item = readdir(dir)) != NULL) {
        int fd = atoi(item->d_name);
        if (fd >= 3) {
          (void)close(fd);
        }
      }

      closedir(dir);

      execl("/bin/sh", "sh", "-c", command, NULL);
      exit(127);
    } else {
      if (read) {
        close(fd_read[1]);
        fcntl(fd_read[0], F_SETFD, FD_CLOEXEC);
        fp_read = fdopen(fd_read[0], "r");
        PADDLE_ENFORCE_NE(
            fp_read,
            nullptr,
            common::errors::External(
                "Failed to open file descriptor via fdopen in AfsManager."));
      }

      if (write) {
        close(fd_write[0]);
        fcntl(fd_write[1], F_SETFD, FD_CLOEXEC);
        fp_write = fdopen(fd_write[1], "w");
        PADDLE_ENFORCE_NE(
            fp_write,
            nullptr,
            common::errors::External(
                "Failed to open file descriptor via fdopen in AfsManager."));
      }
      return 0;
    }
  }
  std::shared_ptr<FILE> GetFile(const std::string& path,
                                const std::string& pipe_command) {
    pid_t pid = 0;
    FILE* wfp = NULL;
    FILE* rfp = NULL;

    // Always use set -eo pipefail. Fail fast and be aware of exit codes.
    std::string cmd = "set -eo pipefail; " + pipe_command;
    int ret =
        PopenBidirectionalInternal(cmd.c_str(), rfp, wfp, pid, true, true);

    PADDLE_ENFORCE_EQ(ret,
                      0,
                      common::errors::PreconditionNotMet(
                          "Called PopenBidirectionalInternal Failed"));
    std::string filename(path);
    if (strncmp(filename.c_str(), "afs:", 4) == 0) {
      filename = filename.substr(4);
    }
    std::thread read_thread(
        &AfsManager::ReadFromAfs, filename, wfp, _afshandler);
    read_thread.detach();
    return {rfp, [pid, cmd](FILE* rfp) {
              int wstatus = -1;
              int ret = -1;
              do {
                ret = waitpid(pid, &wstatus, 0);
              } while (ret == -1 && errno == EINTR);

              fclose(rfp);
              if (wstatus == 0 || wstatus == (128 + SIGPIPE) * 256 ||
                  (wstatus == -1 && errno == ECHILD)) {
                VLOG(3) << "pclose_bidirectional pid[" << pid << "], status["
                        << wstatus << "]";
              } else {
                LOG(WARNING) << "pclose_bidirectional pid[" << pid << "]"
                             << ", ret[" << ret << "] shell open fail";
              }
              if (wstatus == -1 && errno == ECHILD) {
                LOG(WARNING) << "errno is ECHILD";
              }
            }};
  }

 private:
  afs::AfsFileSystem* _afshandler;
  std::mutex g_flock;
};

class BoxWrapper {
 public:
  virtual ~BoxWrapper() {}
  BoxWrapper() {}

  void FeedPass(int date, const std::vector<uint64_t>& feasign_to_box) const;
  void BeginFeedPass(int date, boxps::PSAgentBase** agent) const;
  void EndFeedPass(boxps::PSAgentBase* agent) const;
  void BeginPass() const;
  void EndPass(bool need_save_delta) const;
  void SetTestMode(bool is_test) const;

  template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM = 0>
  void PullSparseCase(const phi::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<float*>& values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size,
                      const int expand_embed_dim);

  void PullSparse(const phi::Place& place,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size,
                  const int expand_embed_dim);

  template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM = 0>
  void PushSparseGradCase(const phi::Place& place,
                          const std::vector<const uint64_t*>& keys,
                          const std::vector<const float*>& grad_values,
                          const std::vector<int64_t>& slot_lengths,
                          const int hidden_size,
                          const int expand_embed_dim,
                          const int batch_size);

  void PushSparseGrad(const phi::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size,
                      const int expand_embed_dim,
                      const int batch_size);

  void CopyForPull(const phi::Place& place,
                   uint64_t** gpu_keys,
                   const std::vector<float*>& values,
                   void* total_values_gpu,
                   const int64_t* gpu_len,
                   const int slot_num,
                   const int hidden_size,
                   const int expand_embed_dim,
                   const int64_t total_length);

  void CopyForPush(const phi::Place& place,
                   const std::vector<const float*>& grad_values,
                   void* total_grad_values_gpu,
                   const std::vector<int64_t>& slot_lengths,
                   const int hidden_size,
                   const int expand_embed_dim,
                   const int64_t total_length,
                   const int batch_size);

  void CopyKeys(const phi::Place& place,
                uint64_t** origin_keys,
                uint64_t* total_keys,
                const int64_t* gpu_len,
                int slot_num,
                int total_len);

  void CheckEmbedSizeIsValid(int embedx_dim, int expand_embed_dim);

  boxps::PSAgentBase* GetAgent() { return p_agent_; }
  void InitializeGPUAndLoadModel(
      const char* conf_file,
      const std::vector<int>& slot_vector,
      const std::vector<std::string>& slot_omit_in_feedpass,
      const std::string& model_path) {
    if (nullptr != s_instance_) {
      VLOG(3) << "Begin InitializeGPU";
      std::vector<gpuStream_t*> stream_list;
      for (int i = 0; i < platform::GetGPUDeviceCount(); ++i) {
        VLOG(3) << "before get context i[" << i << "]";
        phi::GPUContext* context = dynamic_cast<phi::GPUContext*>(
            phi::DeviceContextPool::Instance().Get(phi::GPUPlace(i)));
        stream_list_[i] = context->stream();
        stream_list.push_back(&stream_list_[i]);
      }
      VLOG(2) << "Begin call InitializeGPU in BoxPS";
      // the second parameter is useless
      s_instance_->boxps_ptr_->InitializeGPUAndLoadModel(
          conf_file, -1, stream_list, slot_vector, model_path);
      p_agent_ = boxps::PSAgentBase::GetIns(feedpass_thread_num_);
      p_agent_->Init();
      for (const auto& slot_name : slot_omit_in_feedpass) {
        slot_name_omitted_in_feedpass_.insert(slot_name);
      }
      slot_vector_ = slot_vector;
      keys_tensor.resize(platform::GetGPUDeviceCount());
    }
  }

  int GetFeedpassThreadNum() const { return feedpass_thread_num_; }

  void Finalize() {
    VLOG(3) << "Begin Finalize";
    if (nullptr != s_instance_) {
      s_instance_->boxps_ptr_->Finalize();
    }
  }

  const std::string SaveBase(const char* batch_model_path,
                             const char* xbox_model_path,
                             const std::string& date) {
    VLOG(3) << "Begin SaveBase";
    PADDLE_ENFORCE_EQ(
        date.length(),
        8,
        common::errors::PreconditionNotMet(
            "date[%s] is invalid, correct example is 20190817", date.c_str()));
    int year = std::stoi(date.substr(0, 4));
    int month = std::stoi(date.substr(4, 2));
    int day = std::stoi(date.substr(6, 2));

    struct std::tm b;
    b.tm_year = year - 1900;
    b.tm_mon = month - 1;
    b.tm_mday = day;
    b.tm_min = b.tm_hour = b.tm_sec = 0;
    std::time_t seconds_from_1970 = std::mktime(&b);

    std::string ret_str;
    int ret = boxps_ptr_->SaveBase(
        batch_model_path, xbox_model_path, ret_str, seconds_from_1970 / 86400);
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        common::errors::PreconditionNotMet("SaveBase failed in BoxPS."));
    return ret_str;
  }

  const std::string SaveDelta(const char* xbox_model_path) {
    VLOG(3) << "Begin SaveDelta";
    std::string ret_str;
    int ret = boxps_ptr_->SaveDelta(xbox_model_path, ret_str);
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        common::errors::PreconditionNotMet("SaveDelta failed in BoxPS."));
    return ret_str;
  }

  static std::shared_ptr<BoxWrapper> GetInstance() {
    PADDLE_ENFORCE_EQ(
        s_instance_ == nullptr,
        false,
        common::errors::PreconditionNotMet(
            "GetInstance failed in BoxPs, you should use SetInstance firstly"));
    return s_instance_;
  }

  static std::shared_ptr<BoxWrapper> SetInstance(int embedx_dim = 8,
                                                 int expand_embed_dim = 0) {
    if (nullptr == s_instance_) {
      // If main thread is guaranteed to init this, this lock can be removed
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      if (nullptr == s_instance_) {
        VLOG(3) << "s_instance_ is null";
        s_instance_.reset(new paddle::framework::BoxWrapper());
        s_instance_->boxps_ptr_.reset(
            boxps::BoxPSBase::GetIns(embedx_dim, expand_embed_dim));
        embedx_dim_ = embedx_dim;
        expand_embed_dim_ = expand_embed_dim;
      }
    } else {
      LOG(WARNING) << "You have already used SetInstance() before";
    }
    return s_instance_;
  }

  void InitAfsAPI(const std::string& fs_name,
                  const std::string& fs_ugi,
                  const std::string& conf_path) {
    afs_manager = new AfsManager(fs_name, fs_ugi, conf_path);
    use_afs_api_ = true;
  }

  bool UseAfsApi() const { return use_afs_api_; }

  const std::unordered_set<std::string>& GetOmittedSlot() const {
    return slot_name_omitted_in_feedpass_;
  }

  class MetricMsg {
   public:
    MetricMsg() {}
    MetricMsg(const std::string& label_varname,
              const std::string& pred_varname,
              int metric_phase,
              int bucket_size = 1000000)
        : label_varname_(label_varname),
          pred_varname_(pred_varname),
          metric_phase_(metric_phase) {
      calculator = new BasicAucCalculator();
      calculator->init(bucket_size);
    }
    virtual ~MetricMsg() {}

    int MetricPhase() const { return metric_phase_; }
    BasicAucCalculator* GetCalculator() { return calculator; }
    virtual void add_data(const Scope* exe_scope) {
      std::vector<int64_t> label_data;
      get_data<int64_t>(exe_scope, label_varname_, &label_data);
      std::vector<float> pred_data;
      get_data<float>(exe_scope, pred_varname_, &pred_data);
      auto cal = GetCalculator();
      auto batch_size = label_data.size();
      for (size_t i = 0; i < batch_size; ++i) {
        cal->add_data(pred_data[i], label_data[i]);
      }
    }
    template <class T = float>
    static void get_data(const Scope* exe_scope,
                         const std::string& varname,
                         std::vector<T>* data) {
      auto* var = exe_scope->FindVar(varname.c_str());
      PADDLE_ENFORCE_NOT_NULL(
          var,
          common::errors::NotFound("Error: var %s is not found in scope.",
                                   varname.c_str()));
      auto& gpu_tensor = var->Get<phi::DenseTensor>();
      auto* gpu_data = gpu_tensor.data<T>();
      auto len = gpu_tensor.numel();
      data->resize(len);
#ifdef PADDLE_WITH_HIP
      hipMemcpy(data->data(), gpu_data, sizeof(T) * len, hipMemcpyDeviceToHost);
#else
      cudaMemcpy(
          data->data(), gpu_data, sizeof(T) * len, cudaMemcpyDeviceToHost);
#endif
    }
    static inline std::pair<int, int> parse_cmatch_rank(uint64_t x) {
      // first 32 bit store cmatch and second 32 bit store rank
      return std::make_pair(static_cast<int>(x >> 32),
                            static_cast<int>(x & 0xff));
    }

   protected:
    std::string label_varname_;
    std::string pred_varname_;
    int metric_phase_;
    BasicAucCalculator* calculator;
  };

  class MultiTaskMetricMsg : public MetricMsg {
   public:
    MultiTaskMetricMsg(const std::string& label_varname,
                       const std::string& pred_varname_list,
                       int metric_phase,
                       const std::string& cmatch_rank_group,
                       const std::string& cmatch_rank_varname,
                       int bucket_size = 1000000) {
      label_varname_ = label_varname;
      cmatch_rank_varname_ = cmatch_rank_varname;
      metric_phase_ = metric_phase;
      calculator = new BasicAucCalculator();
      calculator->init(bucket_size);
      for (auto& cmatch_rank : string::split_string(cmatch_rank_group)) {
        const std::vector<std::string>& cur_cmatch_rank =
            string::split_string(cmatch_rank, "_");
        PADDLE_ENFORCE_EQ(
            cur_cmatch_rank.size(),
            2,
            common::errors::PreconditionNotMet("illegal multitask auc spec: %s",
                                               cmatch_rank.c_str()));
        cmatch_rank_v.emplace_back(atoi(cur_cmatch_rank[0].c_str()),
                                   atoi(cur_cmatch_rank[1].c_str()));
      }
      for (const auto& pred_varname : string::split_string(pred_varname_list)) {
        pred_v.emplace_back(pred_varname);
      }
      PADDLE_ENFORCE_EQ(cmatch_rank_v.size(),
                        pred_v.size(),
                        common::errors::PreconditionNotMet(
                            "cmatch_rank's size [%lu] should be equal to pred "
                            "list's size [%lu], but they are not equal",
                            cmatch_rank_v.size(),
                            pred_v.size()));
    }
    virtual ~MultiTaskMetricMsg() {}
    void add_data(const Scope* exe_scope) override {
      std::vector<int64_t> cmatch_rank_data;
      get_data<int64_t>(exe_scope, cmatch_rank_varname_, &cmatch_rank_data);
      std::vector<int64_t> label_data;
      get_data<int64_t>(exe_scope, label_varname_, &label_data);
      size_t batch_size = cmatch_rank_data.size();
      PADDLE_ENFORCE_EQ(
          batch_size,
          label_data.size(),
          common::errors::PreconditionNotMet(
              "illegal batch size: batch_size[%lu] and label_data[%lu]",
              batch_size,
              label_data.size()));

      std::vector<std::vector<float>> pred_data_list(pred_v.size());
      for (size_t i = 0; i < pred_v.size(); ++i) {
        get_data<float>(exe_scope, pred_v[i], &pred_data_list[i]);
      }
      for (size_t i = 0; i < pred_data_list.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            batch_size,
            pred_data_list[i].size(),
            common::errors::PreconditionNotMet(
                "illegal batch size: batch_size[%lu] and pred_data[%lu]",
                batch_size,
                pred_data_list[i].size()));
      }
      auto cal = GetCalculator();
      for (size_t i = 0; i < batch_size; ++i) {
        auto cmatch_rank_it = std::find(cmatch_rank_v.begin(),
                                        cmatch_rank_v.end(),
                                        parse_cmatch_rank(cmatch_rank_data[i]));
        if (cmatch_rank_it != cmatch_rank_v.end()) {
          cal->add_data(pred_data_list[std::distance(cmatch_rank_v.begin(),
                                                     cmatch_rank_it)][i],
                        label_data[i]);
        }
      }
    }

   protected:
    std::vector<std::pair<int, int>> cmatch_rank_v;
    std::vector<std::string> pred_v;
    std::string cmatch_rank_varname_;
  };
  class CmatchRankMetricMsg : public MetricMsg {
   public:
    CmatchRankMetricMsg(const std::string& label_varname,
                        const std::string& pred_varname,
                        int metric_phase,
                        const std::string& cmatch_rank_group,
                        const std::string& cmatch_rank_varname,
                        bool ignore_rank = false,
                        int bucket_size = 1000000) {
      label_varname_ = label_varname;
      pred_varname_ = pred_varname;
      cmatch_rank_varname_ = cmatch_rank_varname;
      metric_phase_ = metric_phase;
      ignore_rank_ = ignore_rank;
      calculator = new BasicAucCalculator();
      calculator->init(bucket_size);
      for (auto& cmatch_rank : string::split_string(cmatch_rank_group)) {
        if (ignore_rank) {  // CmatchAUC
          cmatch_rank_v.emplace_back(atoi(cmatch_rank.c_str()), 0);
          continue;
        }
        const std::vector<std::string>& cur_cmatch_rank =
            string::split_string(cmatch_rank, "_");
        PADDLE_ENFORCE_EQ(
            cur_cmatch_rank.size(),
            2,
            common::errors::PreconditionNotMet(
                "illegal cmatch_rank auc spec: %s", cmatch_rank.c_str()));
        cmatch_rank_v.emplace_back(atoi(cur_cmatch_rank[0].c_str()),
                                   atoi(cur_cmatch_rank[1].c_str()));
      }
    }
    virtual ~CmatchRankMetricMsg() {}
    void add_data(const Scope* exe_scope) override {
      std::vector<int64_t> cmatch_rank_data;
      get_data<int64_t>(exe_scope, cmatch_rank_varname_, &cmatch_rank_data);
      std::vector<int64_t> label_data;
      get_data<int64_t>(exe_scope, label_varname_, &label_data);
      std::vector<float> pred_data;
      get_data<float>(exe_scope, pred_varname_, &pred_data);
      size_t batch_size = cmatch_rank_data.size();
      PADDLE_ENFORCE_EQ(
          batch_size,
          label_data.size(),
          common::errors::PreconditionNotMet(
              "illegal batch size: cmatch_rank[%lu] and label_data[%lu]",
              batch_size,
              label_data.size()));
      PADDLE_ENFORCE_EQ(
          batch_size,
          pred_data.size(),
          common::errors::PreconditionNotMet(
              "illegal batch size: cmatch_rank[%lu] and pred_data[%lu]",
              batch_size,
              pred_data.size()));
      auto cal = GetCalculator();
      for (size_t i = 0; i < batch_size; ++i) {
        const auto& cur_cmatch_rank = parse_cmatch_rank(cmatch_rank_data[i]);
        for (size_t j = 0; j < cmatch_rank_v.size(); ++j) {
          bool is_matched = false;
          if (ignore_rank_) {
            is_matched = cmatch_rank_v[j].first == cur_cmatch_rank.first;
          } else {
            is_matched = cmatch_rank_v[j] == cur_cmatch_rank;
          }
          if (is_matched) {
            cal->add_data(pred_data[i], label_data[i]);
            break;
          }
        }
      }
    }

   protected:
    std::vector<std::pair<int, int>> cmatch_rank_v;
    std::string cmatch_rank_varname_;
    bool ignore_rank_;
  };
  class MaskMetricMsg : public MetricMsg {
   public:
    MaskMetricMsg(const std::string& label_varname,
                  const std::string& pred_varname,
                  int metric_phase,
                  const std::string& mask_varname,
                  int bucket_size = 1000000) {
      label_varname_ = label_varname;
      pred_varname_ = pred_varname;
      mask_varname_ = mask_varname;
      metric_phase_ = metric_phase;
      calculator = new BasicAucCalculator();
      calculator->init(bucket_size);
    }
    virtual ~MaskMetricMsg() {}
    void add_data(const Scope* exe_scope) override {
      std::vector<int64_t> label_data;
      get_data<int64_t>(exe_scope, label_varname_, &label_data);
      std::vector<float> pred_data;
      get_data<float>(exe_scope, pred_varname_, &pred_data);
      std::vector<int64_t> mask_data;
      get_data<int64_t>(exe_scope, mask_varname_, &mask_data);
      auto cal = GetCalculator();
      auto batch_size = label_data.size();
      for (size_t i = 0; i < batch_size; ++i) {
        if (mask_data[i] == 1) {
          cal->add_data(pred_data[i], label_data[i]);
        }
      }
    }

   protected:
    std::string mask_varname_;
  };
  const std::vector<std::string> GetMetricNameList(
      int metric_phase = -1) const {
    VLOG(0) << "Want to Get metric phase: " << metric_phase;
    if (metric_phase == -1) {
      return metric_name_list_;
    } else {
      std::vector<std::string> ret;
      for (const auto& name : metric_name_list_) {
        const auto iter = metric_lists_.find(name);
        PADDLE_ENFORCE_NE(
            iter,
            metric_lists_.end(),
            common::errors::InvalidArgument(
                "The metric name you provided is not registered."));

        if (iter->second->MetricPhase() == metric_phase) {
          VLOG(0) << name << "'s phase is " << iter->second->MetricPhase()
                  << ", we want";
          ret.push_back(name);
        } else {
          VLOG(0) << name << "'s phase is " << iter->second->MetricPhase()
                  << ", not we want";
        }
      }
      return ret;
    }
  }
  int Phase() const { return phase_; }
  void FlipPhase() { phase_ = (phase_ + 1) % phase_num_; }
  std::map<std::string, MetricMsg*>& GetMetricList() { return metric_lists_; }

  void InitMetric(const std::string& method,
                  const std::string& name,
                  const std::string& label_varname,
                  const std::string& pred_varname,
                  const std::string& cmatch_rank_varname,
                  const std::string& mask_varname,
                  int metric_phase,
                  const std::string& cmatch_rank_group,
                  bool ignore_rank,
                  int bucket_size = 1000000) {
    if (method == "AucCalculator") {
      metric_lists_.emplace(
          name,
          new MetricMsg(
              label_varname, pred_varname, metric_phase, bucket_size));
    } else if (method == "MultiTaskAucCalculator") {
      metric_lists_.emplace(name,
                            new MultiTaskMetricMsg(label_varname,
                                                   pred_varname,
                                                   metric_phase,
                                                   cmatch_rank_group,
                                                   cmatch_rank_varname,
                                                   bucket_size));
    } else if (method == "CmatchRankAucCalculator") {
      metric_lists_.emplace(name,
                            new CmatchRankMetricMsg(label_varname,
                                                    pred_varname,
                                                    metric_phase,
                                                    cmatch_rank_group,
                                                    cmatch_rank_varname,
                                                    ignore_rank,
                                                    bucket_size));
    } else if (method == "MaskAucCalculator") {
      metric_lists_.emplace(name,
                            new MaskMetricMsg(label_varname,
                                              pred_varname,
                                              metric_phase,
                                              mask_varname,
                                              bucket_size));
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "PaddleBox only support AucCalculator, MultiTaskAucCalculator "
          "CmatchRankAucCalculator and MaskAucCalculator"));
    }
    metric_name_list_.emplace_back(name);
  }

  const std::vector<float> GetMetricMsg(const std::string& name) {
    const auto iter = metric_lists_.find(name);
    PADDLE_ENFORCE_NE(iter,
                      metric_lists_.end(),
                      common::errors::InvalidArgument(
                          "The metric name you provided is not registered."));
    std::vector<float> metric_return_values_(8, 0.0);
    auto* auc_cal_ = iter->second->GetCalculator();
    auc_cal_->calculate_bucket_error();
    auc_cal_->compute();
    metric_return_values_[0] = auc_cal_->auc();
    metric_return_values_[1] = auc_cal_->bucket_error();
    metric_return_values_[2] = auc_cal_->mae();
    metric_return_values_[3] = auc_cal_->rmse();
    metric_return_values_[4] = auc_cal_->actual_ctr();
    metric_return_values_[5] = auc_cal_->predicted_ctr();
    metric_return_values_[6] =
        auc_cal_->actual_ctr() / auc_cal_->predicted_ctr();
    metric_return_values_[7] = auc_cal_->size();
    auc_cal_->reset();
    return metric_return_values_;
  }

 private:
  static gpuStream_t stream_list_[8];
  static std::shared_ptr<boxps::BoxPSBase> boxps_ptr_;
  boxps::PSAgentBase* p_agent_ = nullptr;
  // TODO(hutuxian): magic number, will add a config to specify
  const int feedpass_thread_num_ = 30;  // magic number
  static std::shared_ptr<BoxWrapper> s_instance_;
  std::unordered_set<std::string> slot_name_omitted_in_feedpass_;
  // EMBEDX_DIM and EXPAND_EMBED_DIM
  static int embedx_dim_;
  static int expand_embed_dim_;

  // Metric Related
  int phase_ = 1;
  int phase_num_ = 2;
  std::map<std::string, MetricMsg*> metric_lists_;
  std::vector<std::string> metric_name_list_;
  std::vector<int> slot_vector_;
  std::vector<phi::DenseTensor> keys_tensor;  // Cache for pull_sparse
  bool use_afs_api_ = false;

 public:
  static AfsManager* afs_manager;

  // Auc Runner
 public:
  void InitializeAucRunner(std::vector<std::vector<std::string>> slot_eval,
                           int thread_num,
                           int pool_size,
                           std::vector<std::string> slot_list) {
    mode_ = 1;
    phase_num_ = static_cast<int>(slot_eval.size());
    phase_ = phase_num_ - 1;
    auc_runner_thread_num_ = thread_num;
    pass_done_semi_ = paddle::framework::MakeChannel<int>();
    pass_done_semi_->Put(1);  // Note: At most 1 pipeline in AucRunner
    random_ins_pool_list.resize(thread_num);

    std::unordered_set<std::string> slot_set;
    for (size_t i = 0; i < slot_eval.size(); ++i) {
      for (const auto& slot : slot_eval[i]) {
        slot_set.insert(slot);
      }
    }
    for (size_t i = 0; i < slot_list.size(); ++i) {
      if (slot_set.find(slot_list[i]) != slot_set.end()) {
        slot_index_to_replace_.insert(static_cast<int16_t>(i));
      }
    }
    for (int i = 0; i < auc_runner_thread_num_; ++i) {
      random_ins_pool_list[i].SetSlotIndexToReplace(slot_index_to_replace_);
    }
    VLOG(0) << "AucRunner configuration: thread number[" << thread_num
            << "], pool size[" << pool_size << "], runner_group[" << phase_num_
            << "]";
    VLOG(0) << "Slots that need to be evaluated:";
    for (auto e : slot_index_to_replace_) {
      VLOG(0) << e << ": " << slot_list[e];
    }
  }
  void GetRandomReplace(const std::vector<Record>& pass_data);
  void AddReplaceFeasign(boxps::PSAgentBase* p_agent, int feed_pass_thread_num);
  void GetRandomData(const std::vector<Record>& pass_data,
                     const std::unordered_set<uint16_t>& slots_to_replace,
                     std::vector<Record>* result);
  int Mode() const { return mode_; }

 private:
  int mode_ = 0;  // 0 means train/test 1 means auc_runner
  int auc_runner_thread_num_ = 1;
  bool init_done_ = false;
  paddle::framework::Channel<int> pass_done_semi_;
  std::unordered_set<uint16_t> slot_index_to_replace_;
  std::vector<RecordCandidateList> random_ins_pool_list;
  std::vector<size_t> replace_idx_;
};
#endif

class BoxHelper {
 public:
  explicit BoxHelper(paddle::framework::Dataset* dataset) : dataset_(dataset) {}
  virtual ~BoxHelper() {}

  void SetDate(int year, int month, int day) {
    year_ = year;
    month_ = month;
    day_ = day;
  }
  void BeginPass() {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->BeginPass();
#endif
  }
  void EndPass(bool need_save_delta) {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->EndPass(need_save_delta);
#endif
  }
  void LoadIntoMemory() {
    platform::Timer timer;
    VLOG(3) << "Begin LoadIntoMemory(), dataset[" << dataset_ << "]";
    timer.Start();
    dataset_->LoadIntoMemory();
    timer.Pause();
    VLOG(0) << "download + parse cost: " << timer.ElapsedSec() << "s";

    timer.Start();
    FeedPass();
    timer.Pause();
    VLOG(0) << "FeedPass cost: " << timer.ElapsedSec() << " s";
    VLOG(3) << "End LoadIntoMemory(), dataset[" << dataset_ << "]";
  }
  void PreLoadIntoMemory() {
    dataset_->PreLoadIntoMemory();
    feed_data_thread_.reset(new std::thread([&]() {
      dataset_->WaitPreLoadDone();
      FeedPass();
    }));
    VLOG(3) << "After PreLoadIntoMemory()";
  }
  void WaitFeedPassDone() { feed_data_thread_->join(); }
  void SlotsShuffle(const std::set<std::string>& slots_to_replace) {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    PADDLE_ENFORCE_EQ(box_ptr->Mode(),
                      1,
                      common::errors::PreconditionNotMet(
                          "Should call InitForAucRunner first."));
    box_ptr->FlipPhase();

    std::unordered_set<uint16_t> index_slots;
    dynamic_cast<MultiSlotDataset*>(dataset_)->PreprocessChannel(
        slots_to_replace, index_slots);
    const std::vector<Record>& pass_data =
        dynamic_cast<MultiSlotDataset*>(dataset_)->GetSlotsOriginalData();
    if (!get_random_replace_done_) {
      box_ptr->GetRandomReplace(pass_data);
      get_random_replace_done_ = true;
    }
    std::vector<Record> random_data;
    random_data.resize(pass_data.size());
    box_ptr->GetRandomData(pass_data, index_slots, &random_data);

    auto new_input_channel = paddle::framework::MakeChannel<Record>();
    new_input_channel->Open();
    new_input_channel->Write(std::move(random_data));
    new_input_channel->Close();
    dynamic_cast<MultiSlotDataset*>(dataset_)->SetInputChannel(
        new_input_channel);
#endif
  }
#ifdef PADDLE_WITH_BOX_PS
  // notify boxps to feed this pass feasigns from SSD to memory
  static void FeedPassThread(const std::deque<Record>& t,
                             int begin_index,
                             int end_index,
                             boxps::PSAgentBase* p_agent,
                             const std::unordered_set<int>& index_map,
                             int thread_id) {
    p_agent->AddKey(0ul, thread_id);
    for (auto iter = t.begin() + begin_index; iter != t.begin() + end_index;
         iter++) {
      const auto& ins = *iter;
      const auto& feasign_v = ins.uint64_feasigns_;
      for (const auto feasign : feasign_v) {
        if (index_map.find(feasign.slot()) != index_map.end()) {
          continue;
        }
        p_agent->AddKey(feasign.sign().uint64_feasign_, thread_id);
      }
    }
  }
#endif
  void FeedPass() {
    VLOG(3) << "Begin FeedPass";
#ifdef PADDLE_WITH_BOX_PS
    struct std::tm b;
    b.tm_year = year_ - 1900;
    b.tm_mon = month_ - 1;
    b.tm_mday = day_;
    b.tm_min = b.tm_hour = b.tm_sec = 0;
    std::time_t x = std::mktime(&b);

    auto box_ptr = BoxWrapper::GetInstance();
    auto input_channel_ =
        dynamic_cast<MultiSlotDataset*>(dataset_)->GetInputChannel();
    const std::deque<Record>& pass_data = input_channel_->GetData();

    // get feasigns that FeedPass doesn't need
    const std::unordered_set<std::string>& slot_name_omitted_in_feedpass_ =
        box_ptr->GetOmittedSlot();
    std::unordered_set<int> slot_id_omitted_in_feedpass_;
    const auto& all_readers = dataset_->GetReaders();
    PADDLE_ENFORCE_GT(all_readers.size(),
                      0,
                      common::errors::PreconditionNotMet(
                          "Readers number must be greater than 0."));
    const auto& all_slots_name = all_readers[0]->GetAllSlotAlias();
    for (size_t i = 0; i < all_slots_name.size(); ++i) {
      if (slot_name_omitted_in_feedpass_.find(all_slots_name[i]) !=
          slot_name_omitted_in_feedpass_.end()) {
        slot_id_omitted_in_feedpass_.insert(i);
      }
    }
    const size_t tnum = box_ptr->GetFeedpassThreadNum();
    boxps::PSAgentBase* p_agent = box_ptr->GetAgent();
    VLOG(3) << "Begin call BeginFeedPass in BoxPS";
    box_ptr->BeginFeedPass(x / 86400, &p_agent);

    std::vector<std::thread> threads;
    size_t len = pass_data.size();
    size_t len_per_thread = len / tnum;
    auto remain = len % tnum;
    size_t begin = 0;
    for (size_t i = 0; i < tnum; i++) {
      threads.push_back(
          std::thread(FeedPassThread,
                      std::ref(pass_data),
                      begin,
                      begin + len_per_thread + (i < remain ? 1 : 0),
                      p_agent,
                      std::ref(slot_id_omitted_in_feedpass_),
                      i));
      begin += len_per_thread + (i < remain ? 1 : 0);
    }
    for (size_t i = 0; i < tnum; ++i) {
      threads[i].join();
    }

    if (box_ptr->Mode() == 1) {
      box_ptr->AddReplaceFeasign(p_agent, tnum);
    }
    VLOG(3) << "Begin call EndFeedPass in BoxPS";
    box_ptr->EndFeedPass(p_agent);
#endif
  }

 private:
  Dataset* dataset_;
  std::shared_ptr<std::thread> feed_data_thread_;
  int year_;
  int month_;
  int day_;
  bool get_random_replace_done_ = false;
};

}  // namespace framework
}  // namespace paddle

#include "paddle/fluid/framework/fleet/box_wrapper_impl.h"
