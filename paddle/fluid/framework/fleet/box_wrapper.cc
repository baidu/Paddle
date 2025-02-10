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

#ifdef PADDLE_WITH_BOX_PS
#include "paddle/fluid/framework/fleet/box_wrapper.h"

#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle::framework {

std::shared_ptr<BoxWrapper> BoxWrapper::s_instance_ = nullptr;
gpuStream_t BoxWrapper::stream_list_[8];
std::shared_ptr<boxps::BoxPSBase> BoxWrapper::boxps_ptr_ = nullptr;
AfsManager* BoxWrapper::afs_manager = nullptr;
int BoxWrapper::embedx_dim_ = 8;
int BoxWrapper::expand_embed_dim_ = 0;

void BasicAucCalculator::compute() {
  double* table[2] = {&_table[0][0], &_table[1][0]};

  double area = 0;
  double fp = 0;
  double tp = 0;

  for (int i = _table_size - 1; i >= 0; i--) {
    double newfp = fp + table[0][i];
    double newtp = tp + table[1][i];
    area += (newfp - fp) * (tp + newtp) / 2;
    fp = newfp;
    tp = newtp;
  }

  if (fp < 1e-3 || tp < 1e-3) {
    _auc = -0.5;  // which means all nonclick or click
  } else {
    _auc = area / (fp * tp);
  }

  _mae = _local_abserr / (fp + tp);
  _rmse = sqrt(_local_sqrerr / (fp + tp));
  _actual_ctr = tp / (fp + tp);
  _predicted_ctr = _local_pred / (fp + tp);
  _size = fp + tp;
}

void BoxWrapper::CheckEmbedSizeIsValid(int embedx_dim, int expand_embed_dim) {
  PADDLE_ENFORCE_EQ(
      embedx_dim_,
      embedx_dim,
      common::errors::InvalidArgument("SetInstance(): invalid embedx_dim. "
                                      "When embedx_dim = %d, but got %d.",
                                      embedx_dim_,
                                      embedx_dim));
  PADDLE_ENFORCE_EQ(expand_embed_dim_,
                    expand_embed_dim,
                    common::errors::InvalidArgument(
                        "SetInstance(): invalid expand_embed_dim. When "
                        "expand_embed_dim = %d, but got %d.",
                        expand_embed_dim_,
                        expand_embed_dim));
}

void BoxWrapper::PullSparse(const phi::Place& place,
                            const std::vector<const uint64_t*>& keys,
                            const std::vector<float*>& values,
                            const std::vector<int64_t>& slot_lengths,
                            const int hidden_size,
                            const int expand_embed_dim) {
#define EMBEDX_CASE(i, ...)                                                    \
  case i: {                                                                    \
    constexpr size_t EmbedxDim = i;                                            \
    switch (expand_embed_dim) {                                                \
      __VA_ARGS__                                                              \
      default:                                                                 \
        PADDLE_THROW(common::errors::InvalidArgument(                          \
            "Unsupported this expand embedding size [%d]", expand_embed_dim)); \
    }                                                                          \
  } break

#define PULLSPARSE_CASE(i, ...)                                            \
  case i: {                                                                \
    constexpr size_t ExpandDim = i;                                        \
    PullSparseCase<EmbedxDim, ExpandDim>(                                  \
        place, keys, values, slot_lengths, hidden_size, expand_embed_dim); \
  } break

  CheckEmbedSizeIsValid(hidden_size - 3, expand_embed_dim);
  switch (hidden_size - 3) {
    EMBEDX_CASE(8, PULLSPARSE_CASE(0); PULLSPARSE_CASE(8);
                PULLSPARSE_CASE(64););
    EMBEDX_CASE(16, PULLSPARSE_CASE(0););
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported this embedding size [%d]", hidden_size - 3));
  }
#undef PULLSPARSE_CASE
#undef EMBEDX_CASE
}

void BoxWrapper::PushSparseGrad(const phi::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<const float*>& grad_values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size,
                                const int expand_embed_dim,
                                const int batch_size) {
#define EMBEDX_CASE(i, ...)                                                    \
  case i: {                                                                    \
    constexpr size_t EmbedxDim = i;                                            \
    switch (expand_embed_dim) {                                                \
      __VA_ARGS__                                                              \
      default:                                                                 \
        PADDLE_THROW(common::errors::InvalidArgument(                          \
            "Unsupported this expand embedding size [%d]", expand_embed_dim)); \
    }                                                                          \
  } break

#define PUSHSPARSE_CASE(i, ...)                                \
  case i: {                                                    \
    constexpr size_t ExpandDim = i;                            \
    PushSparseGradCase<EmbedxDim, ExpandDim>(place,            \
                                             keys,             \
                                             grad_values,      \
                                             slot_lengths,     \
                                             hidden_size,      \
                                             expand_embed_dim, \
                                             batch_size);      \
  } break

  CheckEmbedSizeIsValid(hidden_size - 3, expand_embed_dim);
  switch (hidden_size - 3) {
    EMBEDX_CASE(8, PUSHSPARSE_CASE(0); PUSHSPARSE_CASE(8);
                PUSHSPARSE_CASE(64););
    EMBEDX_CASE(16, PUSHSPARSE_CASE(0););
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported this embedding size [%d]", hidden_size - 3));
  }
#undef PUSHSPARSE_CASE
#undef EMBEDX_CASE
}

void BasicAucCalculator::calculate_bucket_error() {
  double last_ctr = -1;
  double impression_sum = 0;
  double ctr_sum = 0.0;
  double click_sum = 0.0;
  double error_sum = 0.0;
  double error_count = 0;
  double* table[2] = {&_table[0][0], &_table[1][0]};
  for (int i = 0; i < _table_size; i++) {
    double click = table[1][i];
    double show = table[0][i] + table[1][i];
    double ctr = static_cast<double>(i) / _table_size;
    if (fabs(ctr - last_ctr) > kMaxSpan) {
      last_ctr = ctr;
      impression_sum = 0.0;
      ctr_sum = 0.0;
      click_sum = 0.0;
    }
    impression_sum += show;
    ctr_sum += ctr * show;
    click_sum += click;
    double adjust_ctr = ctr_sum / impression_sum;
    double relative_error =
        sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum));
    if (relative_error < kRelativeErrorBound) {
      double actual_ctr = click_sum / impression_sum;
      double relative_ctr_error = fabs(actual_ctr / adjust_ctr - 1);
      error_sum += relative_ctr_error * impression_sum;
      error_count += impression_sum;
      last_ctr = -1;
    }
  }
  _bucket_error = error_count > 0 ? error_sum / error_count : 0.0;
}

// Deprecated: should use BeginFeedPass & EndFeedPass
void BoxWrapper::FeedPass(int date,
                          const std::vector<uint64_t>& feasign_to_box) const {
  int ret = boxps_ptr_->FeedPass(date, feasign_to_box);
  PADDLE_ENFORCE_EQ(
      ret, 0, common::errors::PreconditionNotMet("FeedPass failed in BoxPS."));
}

void BoxWrapper::BeginFeedPass(int date, boxps::PSAgentBase** agent) const {
  int ret = boxps_ptr_->BeginFeedPass(date, *agent);
  PADDLE_ENFORCE_EQ(
      ret,
      0,
      common::errors::PreconditionNotMet("BeginFeedPass failed in BoxPS."));
}

void BoxWrapper::EndFeedPass(boxps::PSAgentBase* agent) const {
  int ret = boxps_ptr_->EndFeedPass(agent);
  PADDLE_ENFORCE_EQ(
      ret,
      0,
      common::errors::PreconditionNotMet("EndFeedPass failed in BoxPS."));
}

void BoxWrapper::BeginPass() const {
  int ret = boxps_ptr_->BeginPass();
  PADDLE_ENFORCE_EQ(
      ret, 0, common::errors::PreconditionNotMet("BeginPass failed in BoxPS."));
}

void BoxWrapper::SetTestMode(bool is_test) const {
  boxps_ptr_->SetTestMode(is_test);
}

void BoxWrapper::EndPass(bool need_save_delta) const {
  int ret = boxps_ptr_->EndPass(need_save_delta);
  PADDLE_ENFORCE_EQ(
      ret, 0, common::errors::PreconditionNotMet("EndPass failed in BoxPS."));
}

void BoxWrapper::GetRandomReplace(const std::vector<Record>& pass_data) {
  VLOG(0) << "Begin GetRandomReplace";
  size_t ins_num = pass_data.size();
  replace_idx_.resize(ins_num);
  for (auto& cand_list : random_ins_pool_list) {
    cand_list.ReInitPass();
  }
  std::vector<std::thread> threads;
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads.push_back(std::thread([this, &pass_data, tid, ins_num]() {
      int start = tid * ins_num / auc_runner_thread_num_;
      int end = (tid + 1) * ins_num / auc_runner_thread_num_;
      VLOG(3) << "GetRandomReplace begin for thread[" << tid
              << "], and process [" << start << ", " << end
              << "), total ins: " << ins_num;
      auto& random_pool = random_ins_pool_list[tid];
      for (int i = start; i < end; ++i) {
        const auto& ins = pass_data[i];
        random_pool.AddAndGet(ins, replace_idx_[i]);
      }
    }));
  }
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads[tid].join();
  }
  pass_done_semi_->Put(1);
  VLOG(0) << "End GetRandomReplace";
}

void BoxWrapper::GetRandomData(
    const std::vector<Record>& pass_data,
    const std::unordered_set<uint16_t>& slots_to_replace,
    std::vector<Record>* result) {
  VLOG(0) << "Begin GetRandomData";
  std::vector<std::thread> threads;
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads.push_back(
        std::thread([this, &pass_data, tid, &slots_to_replace, result]() {
          int debug_erase_cnt = 0;
          int debug_push_cnt = 0;
          size_t ins_num = pass_data.size();
          int start = tid * ins_num / auc_runner_thread_num_;
          int end = (tid + 1) * ins_num / auc_runner_thread_num_;
          VLOG(3) << "GetRandomData begin for thread[" << tid
                  << "], and process [" << start << ", " << end
                  << "), total ins: " << ins_num;
          const auto& random_pool = random_ins_pool_list[tid];
          for (int i = start; i < end; ++i) {
            const auto& ins = pass_data[i];
            const RecordCandidate& rand_rec = random_pool.Get(replace_idx_[i]);
            Record new_rec = ins;
            for (auto it = new_rec.uint64_feasigns_.begin();
                 it != new_rec.uint64_feasigns_.end();) {
              if (slots_to_replace.find(it->slot()) != slots_to_replace.end()) {
                it = new_rec.uint64_feasigns_.erase(it);
                debug_erase_cnt += 1;
              } else {
                ++it;
              }
            }
            for (auto slot : slots_to_replace) {
              auto range = rand_rec.feas_.equal_range(slot);
              for (auto it = range.first; it != range.second; ++it) {
                new_rec.uint64_feasigns_.push_back({it->second, it->first});
                debug_push_cnt += 1;
              }
            }
            (*result)[i] = std::move(new_rec);
          }
          VLOG(3) << "thread[" << tid
                  << "]: erase feasign num: " << debug_erase_cnt
                  << " repush feasign num: " << debug_push_cnt;
        }));
  }
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads[tid].join();
  }
  VLOG(0) << "End GetRandomData";
}

void BoxWrapper::AddReplaceFeasign(boxps::PSAgentBase* p_agent,
                                   int feed_pass_thread_num) {
  VLOG(0) << "Enter AddReplaceFeasign Function";
  int semi;
  pass_done_semi_->Get(semi);
  VLOG(0) << "Last Pass had updated random pool done. Begin AddReplaceFeasign";
  std::vector<std::thread> threads;
  for (int tid = 0; tid < feed_pass_thread_num; ++tid) {
    threads.push_back(std::thread([this, tid, p_agent, feed_pass_thread_num]() {
      VLOG(3) << "AddReplaceFeasign begin for thread[" << tid << "]";
      for (size_t pool_id = tid; pool_id < random_ins_pool_list.size();
           pool_id += feed_pass_thread_num) {
        auto& random_pool = random_ins_pool_list[pool_id];
        for (size_t i = 0; i < random_pool.Size(); ++i) {
          auto& ins_candidate = random_pool.Get(i);
          for (const auto& pair : ins_candidate.feas_) {
            p_agent->AddKey(pair.second.uint64_feasign_, tid);
          }
        }
      }
    }));
  }
  for (int tid = 0; tid < feed_pass_thread_num; ++tid) {
    threads[tid].join();
  }
  VLOG(0) << "End AddReplaceFeasign";
}

}  // namespace paddle::framework
#endif
