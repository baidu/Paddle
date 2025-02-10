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

#include <ThreadPool.h>

#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/phi/core/platform/timer.h"
#include "paddle/utils/string/string_helper.h"

#if defined(PADDLE_WITH_GLOO)
#include <gloo/allreduce.h>

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_PSCORE)
namespace paddle {

namespace framework {

class BasicAucCalculator {
 public:
  BasicAucCalculator() {}
  struct WuaucRecord {
    uint64_t uid_;
    int label_;
    float pred_;
  };

  struct WuaucRocData {
    double tp_;
    double fp_;
    double auc_;
  };
  void init(int table_size);
  void init_wuauc(int table_size);
  void reset();
  void reset_records();
  // add single data in CPU with LOCK, deprecated
  void add_unlock_data(double pred, int label);
  void add_uid_unlock_data(double pred, int label, uint64_t uid);
  // add batch data
  void add_data(const float* d_pred,
                const int64_t* d_label,
                int batch_size,
                const phi::Place& place);
  // add mask data
  void add_mask_data(const float* d_pred,
                     const int64_t* d_label,
                     const int64_t* d_mask,
                     int batch_size,
                     const phi::Place& place);
  // add uid data
  void add_uid_data(const float* d_pred,
                    const int64_t* d_label,
                    const int64_t* d_uid,
                    int batch_size,
                    const phi::Place& place);

  void compute();
  void computeWuAuc();
  WuaucRocData computeSingleUserAuc(const std::vector<WuaucRecord>& records);
  int table_size() const { return _table_size; }
  double bucket_error() const { return _bucket_error; }
  double auc() const { return _auc; }
  double uauc() const { return _uauc; }
  double wuauc() const { return _wuauc; }
  double mae() const { return _mae; }
  double actual_ctr() const { return _actual_ctr; }
  double predicted_ctr() const { return _predicted_ctr; }
  double user_cnt() const { return _user_cnt; }
  double size() const { return _size; }
  double rmse() const { return _rmse; }
  std::unordered_set<uint64_t> uid_keys() const { return _uid_keys; }
  // lock and unlock
  std::mutex& table_mutex(void) { return _table_mutex; }

 private:
  void calculate_bucket_error();

 protected:
  double _local_abserr = 0;
  double _local_sqrerr = 0;
  double _local_pred = 0;
  double _auc = 0;
  double _uauc = 0;
  double _wuauc = 0;
  double _mae = 0;
  double _rmse = 0;
  double _actual_ctr = 0;
  double _predicted_ctr = 0;
  double _size;
  double _user_cnt = 0;
  double _bucket_error = 0;
  std::unordered_set<uint64_t> _uid_keys;

 private:
  void set_table_size(int table_size) { _table_size = table_size; }
  int _table_size;
  std::vector<double> _table[2];
  std::vector<WuaucRecord> wuauc_records_;
  static constexpr double kRelativeErrorBound = 0.05;
  static constexpr double kMaxSpan = 0.01;
  std::mutex _table_mutex;
};

class Metric {
 public:
  virtual ~Metric() {}

  Metric() { fprintf(stdout, "init fleet Metric\n"); }

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

    // add_data
    virtual void add_data(const Scope* exe_scope, const phi::Place& place) {
      int label_len = 0;
      const int64_t* label_data = NULL;
      int pred_len = 0;
      const float* pred_data = NULL;
      get_data<int64_t>(exe_scope, label_varname_, &label_data, &label_len);
      get_data<float>(exe_scope, pred_varname_, &pred_data, &pred_len);
      PADDLE_ENFORCE_EQ(label_len,
                        pred_len,
                        common::errors::PreconditionNotMet(
                            "the predict data length should be consistent with "
                            "the label data length"));
      calculator->add_data(pred_data, label_data, label_len, place);
    }

    // get_data
    template <class T = float>
    static void get_data(const Scope* exe_scope,
                         const std::string& varname,
                         const T** data,
                         int* len) {
      auto* var = exe_scope->FindVar(varname.c_str());
      PADDLE_ENFORCE_NOT_NULL(
          var,
          common::errors::NotFound("Error: var %s is not found in scope.",
                                   varname.c_str()));
      auto& cpu_tensor = var->Get<phi::DenseTensor>();
      *data = cpu_tensor.data<T>();
      *len = cpu_tensor.numel();
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
      auto& cpu_tensor = var->Get<phi::DenseTensor>();
      auto* cpu_data = cpu_tensor.data<T>();
      auto len = cpu_tensor.numel();
      data->resize(len);
      memcpy(data->data(), cpu_data, sizeof(T) * len);
    }

    // parse_cmatch_rank
    static inline std::pair<int, int> parse_cmatch_rank(uint64_t x) {
      // only consider ignore_rank=True
      return std::make_pair(static_cast<int>(x), 0);
      // first 32 bit store cmatch and second 32 bit store rank
      // return std::make_pair(static_cast<int>(x >> 32),
      //                       static_cast<int>(x & 0xff));
    }

   protected:
    std::string label_varname_;
    std::string pred_varname_;
    int metric_phase_;
    BasicAucCalculator* calculator;
  };

  class WuAucMetricMsg : public MetricMsg {
   public:
    WuAucMetricMsg(const std::string& label_varname,
                   const std::string& pred_varname,
                   const std::string& uid_varname,
                   int metric_phase,
                   int bucket_size UNUSED = 1000000) {
      label_varname_ = label_varname;
      pred_varname_ = pred_varname;
      uid_varname_ = uid_varname;
      metric_phase_ = metric_phase;
      calculator = new BasicAucCalculator();
    }
    virtual ~WuAucMetricMsg() {}
    void add_data(const Scope* exe_scope, const phi::Place& place) override {
      int label_len = 0;
      const int64_t* label_data = NULL;
      get_data<int64_t>(exe_scope, label_varname_, &label_data, &label_len);

      int pred_len = 0;
      const float* pred_data = NULL;
      get_data<float>(exe_scope, pred_varname_, &pred_data, &pred_len);

      int uid_len = 0;
      const int64_t* uid_data = NULL;
      get_data<int64_t>(exe_scope, uid_varname_, &uid_data, &uid_len);
      PADDLE_ENFORCE_EQ(label_len,
                        uid_len,
                        common::errors::PreconditionNotMet(
                            "the predict data length should be consistent with "
                            "the label data length"));
      auto cal = GetCalculator();
      cal->add_uid_data(pred_data, label_data, uid_data, label_len, place);
    }

   protected:
    std::string uid_varname_;
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
    void add_data(const Scope* exe_scope,
                  const phi::Place& place UNUSED) override {
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
      std::lock_guard<std::mutex> lock(cal->table_mutex());
      for (size_t i = 0; i < batch_size; ++i) {
        auto cmatch_rank_it = std::find(cmatch_rank_v.begin(),
                                        cmatch_rank_v.end(),
                                        parse_cmatch_rank(cmatch_rank_data[i]));
        if (cmatch_rank_it != cmatch_rank_v.end()) {
          cal->add_unlock_data(pred_data_list[std::distance(
                                   cmatch_rank_v.begin(), cmatch_rank_it)][i],
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
    void add_data(const Scope* exe_scope,
                  const phi::Place& place UNUSED) override {
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
      std::lock_guard<std::mutex> lock(cal->table_mutex());
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
            cal->add_unlock_data(pred_data[i], label_data[i]);
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
    void add_data(const Scope* exe_scope, const phi::Place& place) override {
      int label_len = 0;
      const int64_t* label_data = NULL;
      get_data<int64_t>(exe_scope, label_varname_, &label_data, &label_len);

      int pred_len = 0;
      const float* pred_data = NULL;
      get_data<float>(exe_scope, pred_varname_, &pred_data, &pred_len);

      int mask_len = 0;
      const int64_t* mask_data = NULL;
      get_data<int64_t>(exe_scope, mask_varname_, &mask_data, &mask_len);
      PADDLE_ENFORCE_EQ(label_len,
                        mask_len,
                        common::errors::PreconditionNotMet(
                            "the predict data length should be consistent with "
                            "the label data length"));
      auto cal = GetCalculator();
      cal->add_mask_data(pred_data, label_data, mask_data, label_len, place);
    }

   protected:
    std::string mask_varname_;
  };

  class CmatchRankMaskMetricMsg : public MetricMsg {
   public:
    CmatchRankMaskMetricMsg(const std::string& label_varname,
                            const std::string& pred_varname,
                            int metric_phase,
                            const std::string& cmatch_rank_group,
                            const std::string& cmatch_rank_varname,
                            bool ignore_rank = false,
                            const std::string& mask_varname = "",
                            int bucket_size = 1000000) {
      label_varname_ = label_varname;
      pred_varname_ = pred_varname;
      cmatch_rank_varname_ = cmatch_rank_varname;
      metric_phase_ = metric_phase;
      ignore_rank_ = ignore_rank;
      mask_varname_ = mask_varname;
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
    virtual ~CmatchRankMaskMetricMsg() {}
    void add_data(const Scope* exe_scope,
                  const phi::Place& place UNUSED) override {
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

      std::vector<int64_t> mask_data;
      if (!mask_varname_.empty()) {
        get_data<int64_t>(exe_scope, mask_varname_, &mask_data);
        PADDLE_ENFORCE_EQ(
            batch_size,
            mask_data.size(),
            common::errors::PreconditionNotMet(
                "illegal batch size: cmatch_rank[%lu] and mask_data[%lu]",
                batch_size,
                mask_data.size()));
      }

      auto cal = GetCalculator();
      std::lock_guard<std::mutex> lock(cal->table_mutex());
      for (size_t i = 0; i < batch_size; ++i) {
        const auto& cur_cmatch_rank = parse_cmatch_rank(cmatch_rank_data[i]);
        for (size_t j = 0; j < cmatch_rank_v.size(); ++j) {
          if (!mask_data.empty() && !mask_data[i]) {
            continue;
          }
          bool is_matched = false;
          if (ignore_rank_) {
            is_matched = cmatch_rank_v[j].first == cur_cmatch_rank.first;
          } else {
            is_matched = cmatch_rank_v[j] == cur_cmatch_rank;
          }
          if (is_matched) {
            cal->add_unlock_data(pred_data[i], label_data[i]);
            break;
          }
        }
      }
    }

   protected:
    std::vector<std::pair<int, int>> cmatch_rank_v;
    std::string cmatch_rank_varname_;
    bool ignore_rank_;
    std::string mask_varname_;
  };

  static std::shared_ptr<Metric> GetInstance() {
    // PADDLE_ENFORCE_EQ(
    //     s_instance_ == nullptr, false,
    //     common::errors::PreconditionNotMet(
    //         "GetInstance failed in Metric, you should use SetInstance
    //         firstly"));
    return s_instance_;
  }

  static std::shared_ptr<Metric> SetInstance() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (nullptr == s_instance_) {
      VLOG(3) << "s_instance_ is null";
      s_instance_.reset(new paddle::framework::Metric());
    } else {
      LOG(WARNING) << "You have already used SetInstance() before";
    }
    return s_instance_;
  }

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
          VLOG(3) << name << "'s phase is " << iter->second->MetricPhase()
                  << ", we want";
          ret.push_back(name);
        } else {
          VLOG(3) << name << "'s phase is " << iter->second->MetricPhase()
                  << ", not we want";
        }
      }
      return ret;
    }
  }
  int Phase() const { return phase_; }
  int PhaseNum() const { return phase_num_; }
  void FlipPhase() { phase_ = (phase_ + 1) % phase_num_; }
  std::map<std::string, MetricMsg*>& GetMetricList() { return metric_lists_; }

  void InitMetric(const std::string& method,
                  const std::string& name,
                  const std::string& label_varname,
                  const std::string& pred_varname,
                  const std::string& cmatch_rank_varname,
                  const std::string& mask_varname,
                  const std::string& uid_varname,
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
    } else if (method == "CmatchRankMaskAucCalculator") {
      metric_lists_.emplace(name,
                            new CmatchRankMaskMetricMsg(label_varname,
                                                        pred_varname,
                                                        metric_phase,
                                                        cmatch_rank_group,
                                                        cmatch_rank_varname,
                                                        ignore_rank,
                                                        mask_varname,
                                                        bucket_size));
    } else if (method == "WuAucCalculator") {
      metric_lists_.emplace(name,
                            new WuAucMetricMsg(label_varname,
                                               pred_varname,
                                               uid_varname,
                                               metric_phase,
                                               bucket_size));
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "PSLIB Metrics only support AucCalculator, MultiTaskAucCalculator, "
          "CmatchRankAucCalculator, MaskAucCalculator, WuAucCalculator and "
          "CmatchRankMaskAucCalculator"));
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

  const std::vector<float> GetWuAucMetricMsg(const std::string& name) {
    const auto iter = metric_lists_.find(name);
    PADDLE_ENFORCE_NE(iter,
                      metric_lists_.end(),
                      common::errors::InvalidArgument(
                          "The metric name you provided is not registered."));
    VLOG(0) << "begin GetWuAucMetricMsg";
    std::vector<float> metric_return_values_(6, 0.0);
    auto* auc_cal_ = iter->second->GetCalculator();
    auc_cal_->computeWuAuc();
    metric_return_values_[0] = auc_cal_->user_cnt();
    metric_return_values_[1] = auc_cal_->size();
    metric_return_values_[2] = auc_cal_->uauc();
    metric_return_values_[3] = auc_cal_->wuauc();
    metric_return_values_[4] =
        metric_return_values_[2] / (metric_return_values_[0] + 1e-10);
    metric_return_values_[5] =
        metric_return_values_[3] / (metric_return_values_[1] + 1e-10);

#if defined(PADDLE_WITH_GLOO)
    auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
    if (gloo_wrapper->Size() > 1) {
      auto global_metric_return_values_ =
          gloo_wrapper->AllReduce(metric_return_values_, "sum");
      global_metric_return_values_[4] =
          global_metric_return_values_[2] /
          (global_metric_return_values_[0] + 1e-10);
      global_metric_return_values_[5] =
          global_metric_return_values_[3] /
          (global_metric_return_values_[1] + 1e-10);
      auc_cal_->reset_records();
      return global_metric_return_values_;
    } else {
      auc_cal_->reset_records();
      return metric_return_values_;
    }
#else
    auc_cal_->reset_records();
    return metric_return_values_;
#endif
  }

 private:
  static std::shared_ptr<Metric> s_instance_;

  // Metric Related
  int phase_ = 1;
  int phase_num_ = 2;
  std::map<std::string, MetricMsg*> metric_lists_;
  std::vector<std::string> metric_name_list_;
};
}  // namespace framework
}  // namespace paddle
#endif
