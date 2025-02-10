// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <memory>
#include <sstream>

#include <fstream>
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/ir/group_schedule/config/database.h"
#include "paddle/cinn/ir/group_schedule/config/file_database.h"
#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/ir/group_schedule/config/schedule_config_manager.h"
#include "paddle/cinn/ir/group_schedule/search/config_searcher.h"
#include "paddle/cinn/ir/group_schedule/search/measurer.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/performance_statistician.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"

PD_DECLARE_string(tile_config_policy);
PD_DECLARE_string(cinn_tile_config_filename_label);
COMMON_DECLARE_bool(print_ir);
PD_DECLARE_bool(cinn_measure_kernel_time);
PHI_DECLARE_bool(enable_cinn_compile_cache);

#define MKDIR(path) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
struct PerfTestConfig {
  int InnerTestNum;
  int OuterTestNum;
  int MaxTestNum;
  float EarlyStopThreshold;
  PerfTestConfig()
      : InnerTestNum(1),
        OuterTestNum(3),
        MaxTestNum(3),
        EarlyStopThreshold(1.2) {}
  PerfTestConfig(int inner_num, int outer_num, int max_num, float earlystop)
      : InnerTestNum(inner_num),
        OuterTestNum(outer_num),
        MaxTestNum(max_num),
        EarlyStopThreshold(earlystop) {}
};

std::string GenCSVFilePath(const cinn::common::Target target,
                           const cinn::ir::IterSpaceType &iter_space_type) {
  std::string dirname = "";
  std::string filename = "";
  for (auto i : iter_space_type) {
    dirname += i.first;
    dirname += "_";
    filename += i.first + i.second;
    filename += "_";
  }
  const std::string kDirSuffix = "_EREBE";
  dirname = dirname.substr(0, dirname.size() - 1) + kDirSuffix;
  filename = filename.substr(0, filename.size() - 1);

  auto checkexist = [](std::string test_path) {
    bool path_exists = false;
    struct stat statbuf;
    if (stat(test_path.c_str(), &statbuf) != -1) {
      if (S_ISDIR(statbuf.st_mode)) {
        path_exists = true;
      }
    }
    if (!path_exists) {
      PADDLE_ENFORCE_NE(MKDIR(test_path.c_str()),
                        -1,
                        ::common::errors::PreconditionNotMet(
                            "Can not create directory: %s, Make sure you "
                            "have permission to write",
                            test_path));
    }
  };
  std::string root_path = FLAGS_cinn_tile_config_filename_label;
  if (root_path == "") {
    const std::string kTestFileDir = "./tile_file_test/";
    root_path = kTestFileDir;
  }
  std::string target_str = target.arch_str() + "_" + target.device_name_str();
  checkexist(root_path);
  checkexist(root_path + target_str);
  checkexist(root_path + target_str + "/" + dirname);
  VLOG(3) << "Dump_path is " << root_path + dirname + "/" + filename + ".csv";
  return root_path + target_str + "/" + dirname + "/" + filename + ".csv";
}

void WriteBucketInfo(
    std::ofstream &os,
    const std::vector<std::pair<std::string, std::string>> &iter_space_type,
    const cinn::ir::BucketInfo &bucket_info) {
  std::stringstream ss;
  ss << " { ";
  for (int i = 0; i < iter_space_type.size(); ++i) {
    ss << iter_space_type[i].first << "_" << iter_space_type[i].second << ": "
       << bucket_info.space[i].lower_bound << "-"
       << bucket_info.space[i].upper_bound << " ";
  }
  ss << "} ";
  os << ss.str() << " \t ";
}

// reduce_sum
std::shared_ptr<::pir::Program> BuildReduceSumProgram(int spatial_size,
                                                      int reduce_size) {
  ::pir::IrContext *ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const std::vector<int64_t> shape = {spatial_size, reduce_size};
  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   "x", shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto out = builder
                 .Build<paddle::dialect::SumOp>(
                     x, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
                 .result(0);
  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

// layerNorm
std::shared_ptr<::pir::Program> BuildLayerNormProgram(int spatial_size,
                                                      int reduce_size) {
  ::pir::IrContext *ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const std::vector<int64_t> shape = {spatial_size, reduce_size};
  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   "x", shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto sum_val =
      builder
          .Build<paddle::dialect::SumOp>(
              x, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
          .result(0);

  auto divide_num =
      builder
          .Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({1}), 1024, phi::DataType::FLOAT32)
          .out();
  auto mean_val =
      builder.Build<paddle::dialect::DivideOp>(sum_val, divide_num).result(0);
  auto sub_num =
      builder.Build<paddle::dialect::SubtractOp>(x, mean_val).result(0);
  auto pow_val = builder.Build<paddle::dialect::PowOp>(x, 2.0).result(0);
  auto pow_sum =
      builder
          .Build<paddle::dialect::SumOp>(
              pow_val, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
          .result(0);
  auto variance_val =
      builder.Build<paddle::dialect::DivideOp>(pow_sum, divide_num).result(0);
  auto add_num =
      builder
          .Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({1}), 1e-6, phi::DataType::FLOAT32)
          .out();
  auto add_val =
      builder.Build<paddle::dialect::AddOp>(variance_val, add_num).result(0);
  auto rsqrt_val = builder.Build<paddle::dialect::RsqrtOp>(add_val).result(0);
  auto out =
      builder.Build<paddle::dialect::MultiplyOp>(rsqrt_val, sub_num).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

// softmax
std::shared_ptr<::pir::Program> BuildSoftmaxProgram(int spatial_size,
                                                    int reduce_size) {
  ::pir::IrContext *ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const std::vector<int64_t> shape = {spatial_size, reduce_size};
  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   "x", shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto max_num =
      builder.Build<paddle::dialect::MaxOp>(x, std::vector<int64_t>{-1}, true)
          .result(0);
  auto sub_num =
      builder.Build<paddle::dialect::SubtractOp>(x, max_num).result(0);
  auto exp_num = builder.Build<paddle::dialect::ExpOp>(sub_num).result(0);
  auto exp_sum =
      builder
          .Build<paddle::dialect::SumOp>(
              exp_num, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
          .result(0);
  auto out =
      builder.Build<paddle::dialect::DivideOp>(exp_num, exp_sum).result(0);
  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

// RMSNorm
std::shared_ptr<::pir::Program> BuildRMSNormProgram(int spatial_size,
                                                    int reduce_size) {
  ::pir::IrContext *ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const std::vector<int64_t> shape = {spatial_size, reduce_size};
  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   "x", shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto pow_val = builder.Build<paddle::dialect::PowOp>(x, 2.0).result(0);
  auto sum_val =
      builder
          .Build<paddle::dialect::SumOp>(
              pow_val, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
          .result(0);
  auto divide_num =
      builder
          .Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({1}), reduce_size, phi::DataType::FLOAT32)
          .out();
  auto div_val =
      builder.Build<paddle::dialect::DivideOp>(sum_val, divide_num).result(0);
  auto add_num =
      builder
          .Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({1}), 1e-6, phi::DataType::FLOAT32)
          .out();
  auto add_val =
      builder.Build<paddle::dialect::AddOp>(div_val, add_num).result(0);
  auto rsqrt_val = builder.Build<paddle::dialect::RsqrtOp>(add_val).result(0);
  auto out = builder.Build<paddle::dialect::MultiplyOp>(rsqrt_val, x).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

// Build test program for spatial reduce.
std::shared_ptr<::pir::Program> BuildSpatialReduceProgram(int spatial_size,
                                                          int reduce_size) {
  std::shared_ptr<::pir::Program> program;
  program = BuildSoftmaxProgram(spatial_size, reduce_size);
  return program;
}

// Get the tile size configuration for the given dimension lower bound
// dynamically.
int get_tile_size_config_in_small_area(int dimension_lower) {
  if (dimension_lower <= 2) {
    return 126;
  } else if (dimension_lower <= 128) {
    return 384;
  } else if (dimension_lower <= 512) {
    return 512;
  } else if (dimension_lower <= 1024) {
    return 1024;
  } else if (dimension_lower <= 2048) {
    return 2048;
  }
}

int get_tile_size_config_in_large_area(int dimension_lower) {
  if (dimension_lower <= 2) {
    return 510;
  } else if (dimension_lower <= 512) {
    return 512;
  } else if (dimension_lower <= 4096) {
    return 4096;
  } else if (dimension_lower <= 8192) {
    return 8192;
  } else if (dimension_lower <= 16384) {
    return 16384;
  }
}

std::shared_ptr<::pir::Program> BuildProgram(bool is_spatial_dynamic,
                                             bool is_reduce_dynamic,
                                             int s_dimension_lower,
                                             int r_dimension_lower) {
  std::shared_ptr<::pir::Program> program;
  if (!is_spatial_dynamic && !is_reduce_dynamic) {
    program = BuildSpatialReduceProgram(s_dimension_lower, r_dimension_lower);
  } else if (is_spatial_dynamic && !is_reduce_dynamic) {
    program = BuildSpatialReduceProgram(-1, r_dimension_lower);
  } else if (!is_spatial_dynamic && is_reduce_dynamic) {
    program = BuildSpatialReduceProgram(s_dimension_lower, -1);
  } else {
    program = BuildSpatialReduceProgram(-1, -1);
  }
  return program;
}

cinn::ir::BucketInfo CreateBucket(int s_dimension_lower,
                                  int spatial_tile_width,
                                  int r_dimension_lower,
                                  int reduce_tile_width,
                                  bool is_spatial_dynamic,
                                  bool is_reduce_dynamic) {
  cinn::ir::BucketInfo bucket_info;
  bucket_info.space.push_back(cinn::ir::BucketInfo::Dimension{
      s_dimension_lower,
      s_dimension_lower + spatial_tile_width - 1,
      "S",
      /* is_dynamic = */ is_spatial_dynamic});
  bucket_info.space.push_back(
      cinn::ir::BucketInfo::Dimension{r_dimension_lower,
                                      r_dimension_lower + reduce_tile_width - 1,
                                      "R",
                                      /* is_dynamic = */ is_reduce_dynamic});
  return bucket_info;
}

cinn::ir::search::CandidateType GetCandidate(
    const cinn::ir::TileConfigMap &best_tile_config_map,
    int s_dimension_lower,
    int spatial_tile_width,
    int r_dimension_lower,
    int reduce_tile_width) {
  cinn::ir::search::CandidateType best_candidate;
  for (auto &it : best_tile_config_map) {
    auto s_flag = false, r_flag = false;
    auto dims = it.first.space.size();
    // SR type support only
    if (dims == 2) {
      if (it.first.space[0].lower_bound <= s_dimension_lower &&
          it.first.space[0].upper_bound >=
              s_dimension_lower + spatial_tile_width - 1) {
        s_flag = true;
      } else if (it.first.space[0].lower_bound == 4096 &&
                 s_dimension_lower == 4096) {
        s_flag = true;
      }
      if (it.first.space[1].lower_bound <= r_dimension_lower &&
          it.first.space[1].upper_bound >=
              r_dimension_lower + reduce_tile_width - 1) {
        r_flag = true;
      } else if (it.first.space[1].lower_bound == 4096 &&
                 r_dimension_lower == 4096) {
        r_flag = true;
      }
    } else {
      PADDLE_THROW(::common::errors::Unavailable("Now just support SR type."));
    }
    if (s_flag == true && r_flag == true) {
      best_candidate = {it.second.warp_num,
                        it.second.tree_reduce_num,
                        it.second.spatial_inner_num};
      break;
    }
  }
  if (best_candidate.empty()) {
    PADDLE_THROW(
        ::common::errors::Unavailable("Not found the best candidate."));
  }
  return best_candidate;
}

void TestWindowPerformance(
    std::ofstream &os,
    const PerfTestConfig &perf_test_config,
    const int graph_num,
    const int s_dimension_lower,
    const int spatial_tile_width,
    const int r_dimension_lower,
    const int reduce_tile_width,
    const int is_spatial_dynamic,
    const int is_reduce_dynamic,
    const double s_weight,
    const double r_weight,
    const double sampling_prob,
    const int kMaxSamplingTimes,
    const int kRepeats,
    const cinn::ir::TileConfigMap &best_tile_config_map,
    const std::vector<std::pair<std::string, std::string>> &iter_space_type) {
  std::vector<double> s_weights =
      std::vector<double>(spatial_tile_width, s_weight);
  std::vector<double> r_weights =
      std::vector<double>(reduce_tile_width, r_weight);

  LOG(INFO) << "spatial tile dimension lower bound = " << s_dimension_lower
            << ", reduce tile dimension lower bound = " << r_dimension_lower
            << std::endl;

  // Construct pir::Program.
  std::shared_ptr<::pir::Program> program;
  program = BuildProgram(is_spatial_dynamic,
                         is_reduce_dynamic,
                         s_dimension_lower,
                         r_dimension_lower);

  // Construct iter space and objective function.
  cinn::ir::BucketInfo bucket_info = CreateBucket(s_dimension_lower,
                                                  spatial_tile_width,
                                                  r_dimension_lower,
                                                  reduce_tile_width,
                                                  is_spatial_dynamic,
                                                  is_reduce_dynamic);
  std::unique_ptr<cinn::ir::search::BaseObjectiveFunc> obj_func =
      std::make_unique<cinn::ir::search::WeightedSamplingTrailObjectiveFunc>(
          program.get(),
          bucket_info,
          sampling_prob,
          kMaxSamplingTimes,
          kRepeats,
          std::vector<std::vector<double>>{s_weights, r_weights});
  cinn::ir::search::CandidateType best_candidate =
      GetCandidate(best_tile_config_map,
                   s_dimension_lower,
                   spatial_tile_width,
                   r_dimension_lower,
                   reduce_tile_width);

  // Write current bucket info to csv file.
  WriteBucketInfo(os, iter_space_type, bucket_info);

  int InnerTestNum = perf_test_config.InnerTestNum;
  int OuterTestNum = perf_test_config.OuterTestNum;
  int MaxTestNum = perf_test_config.MaxTestNum;
  float EarlyStopThreshold = perf_test_config.EarlyStopThreshold;
  cinn::ir::search::ScoreType record_baseline_score;
  cinn::ir::search::ScoreType record_best_score;
  double record_best_variance = 0;
  // start test loop
  for (int current_test_num = 0; current_test_num < MaxTestNum;
       ++current_test_num) {
    std::vector<cinn::ir::search::ScoreType> vec_best;
    std::vector<cinn::ir::search::ScoreType> vec_baseline;
    // start outer measure loop
    for (int i = 0; i < OuterTestNum; ++i) {
      cinn::ir::search::CandidateType default_candidate;
      std::vector<cinn::ir::search::ScoreType> vec_best_score;
      std::vector<cinn::ir::search::ScoreType> vec_baseline_score;
      // start inner measure loop
      for (int i = 0; i < InnerTestNum; i++) {
        FLAGS_tile_config_policy = "default";
        cinn::ir::search::ScoreType temp_baseline_score =
            (*obj_func)(default_candidate);
        FLAGS_tile_config_policy = "search";
        cinn::ir::search::ScoreType temp_best_score =
            (*obj_func)(best_candidate);
        vec_best_score.push_back(temp_best_score);
        vec_baseline_score.push_back(temp_baseline_score);
      }
      float best_total_time =
          std::accumulate(vec_best_score.begin(), vec_best_score.end(), 0.0);
      float baseline_total_time = std::accumulate(
          vec_baseline_score.begin(), vec_baseline_score.end(), 0.0);
      float best_avg_time = best_total_time / InnerTestNum;
      float baseline_avg_time = baseline_total_time / InnerTestNum;
      vec_best.push_back(best_avg_time);
      vec_baseline.push_back(baseline_avg_time);
    }
    cinn::ir::search::ScoreType best_mean =
        std::accumulate(vec_best.begin(), vec_best.end(), 0.0) / OuterTestNum;
    cinn::ir::search::ScoreType baseline_mean =
        std::accumulate(vec_baseline.begin(), vec_baseline.end(), 0.0) /
        OuterTestNum;

    // Compute variance of OuterTestNum times of outer measure in each test
    double best_variance = 0.0;
    double baseline_variance = 0.0;
    for (int i = 0; i < OuterTestNum; i++) {
      best_variance = best_variance + pow(vec_best[i] - best_mean, 2);
      baseline_variance =
          baseline_variance + pow(vec_baseline[i] - baseline_mean, 2);
    }
    best_variance = pow(best_variance / OuterTestNum, 0.5);
    baseline_variance = pow(baseline_variance / OuterTestNum, 0.5);

    // Early stop when current best_variance smaller than EarlyStopThreshold
    if ((best_variance < EarlyStopThreshold) &&
        (baseline_variance < EarlyStopThreshold)) {
      record_baseline_score = baseline_mean;
      record_best_score = best_mean;
      record_best_variance = best_variance + baseline_variance;
      break;
    } else {
      if (record_best_variance == 0) {
        record_best_variance = best_variance + baseline_variance;
        record_baseline_score = baseline_mean;
        record_best_score = best_mean;
      } else if ((best_variance + baseline_variance) < record_best_variance) {
        record_best_variance = best_variance + baseline_variance;
        record_baseline_score = baseline_mean;
        record_best_score = best_mean;
      }
    }
  }  // end of test_num loop
  cinn::ir::search::ScoreType optim_percentage =
      (1 / (record_best_score)-1 / record_baseline_score) *
      record_baseline_score;
  LOG(INFO) << "Best score: " << record_best_score / graph_num;
  LOG(INFO) << "Baseline score: " << record_baseline_score / graph_num;
  LOG(INFO) << "variance: " << (record_best_variance / 2);
  LOG(INFO) << "optim percentage: " << optim_percentage;
  LOG(INFO) << "Best candidate: " << best_candidate[0] << " "
            << best_candidate[1] << " " << best_candidate[2];
  // Write measure results to csv file
  os << std::setprecision(3) << "  " << record_baseline_score / graph_num
     << " \t " << record_best_score / graph_num << " \t " << optim_percentage
     << "\n";
}

void TestPerformanceForTileConfig(int spatial_left_bound,
                                  int spatial_right_bound,
                                  int reduce_left_bound,
                                  int reduce_right_bound,
                                  bool is_spatial_dynamic,
                                  bool is_reduce_dynamic,
                                  bool test_single_large) {
  FLAGS_enable_cinn_compile_cache = false;
  FLAGS_cinn_measure_kernel_time = true;
  // set tile_file path to test path when user use default setting
  std::string root_path = FLAGS_cinn_tile_config_filename_label;
  if (root_path == "") {
    const std::string kTestFileDir = "./tile_file_test/";
    FLAGS_cinn_tile_config_filename_label = kTestFileDir;
  }

  constexpr int kThreadsPerWarp = 32;
  constexpr int kMaxThreadsPerBlock = 1024;

  // now each has the same weight
  constexpr double s_w = 0.05;
  constexpr double r_w = 0.05;
  constexpr double sampling_prob = 1.0;
  constexpr int kMaxSamplingTimes = 360;
  constexpr int kRepeats = 5;
  constexpr int kInnerTestNum = 1;
  constexpr int kOuterTestNum = 3;
  constexpr int kMaxTestNum = 1;
  constexpr float kEarlyStopThreshold = 1.2;
  // number of nodes in cudaGraph for test, which is defined in
  // performance_statistician.h as graph_nodes_num_. This parameter is set to
  // make measure results corresponding to one launch for better readability
  constexpr int kGraphNum = 25;

  // Define the initial grid size for the spatial and reduction dimensions
  int spatial_tile_config = 0, reduce_tile_config = 0;
  int spatial_tile_width = 0, reduce_tile_width = 0;
  // Define weight for each dimension
  double s_weight = (is_spatial_dynamic ? s_w : 1.0);
  double r_weight = (is_reduce_dynamic ? r_w : 1.0);

  auto s_dimension_type = "S";
  auto r_dimension_type = "R";

  // Define the performance test configuration.
  PerfTestConfig perf_test_config = {
      kInnerTestNum, kOuterTestNum, kMaxTestNum, kEarlyStopThreshold};

  // Get best configuration from json by file database.
  std::vector<std::pair<std::string, std::string>> iter_space_type = {
      std::make_pair(s_dimension_type,
                     is_spatial_dynamic == true ? "dynamic" : "static"),
      std::make_pair(r_dimension_type,
                     is_reduce_dynamic == true ? "dynamic" : "static")};
  cinn::ir::FileTileConfigDatabase file_database;
  cinn::ir::TileConfigMap best_tile_config_map =
      file_database.GetConfigs(cinn::common::DefaultTarget(), iter_space_type);

  // build csv file
  std::string dump_path =
      GenCSVFilePath(cinn::common::DefaultTarget(), iter_space_type);
  std::ofstream os(dump_path, std::ofstream::app);
  PADDLE_ENFORCE_EQ(os.good(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Cannot open the file to write:  %s", dump_path));
  // run performance test for each data grid
  for (int s_dimension_lower = spatial_left_bound;
       s_dimension_lower < spatial_right_bound ||
       s_dimension_lower == spatial_right_bound &&
           spatial_left_bound == spatial_right_bound;
       s_dimension_lower += spatial_tile_config) {
    // adjust the tile size for the spatial dimension dymaically
    spatial_tile_config = get_tile_size_config_in_small_area(s_dimension_lower);
    spatial_tile_width = (is_spatial_dynamic ? spatial_tile_config : 1);
    for (int r_dimension_lower = reduce_left_bound;
         r_dimension_lower < reduce_right_bound ||
         r_dimension_lower == reduce_right_bound &&
             reduce_left_bound == reduce_right_bound;
         r_dimension_lower += reduce_tile_config) {
      // adjust the tile size for the reduce dimension dymaically
      reduce_tile_config =
          get_tile_size_config_in_small_area(r_dimension_lower);
      reduce_tile_width = (is_reduce_dynamic ? reduce_tile_config : 1);
      TestWindowPerformance(os,
                            perf_test_config,
                            kGraphNum,
                            s_dimension_lower,
                            spatial_tile_width,
                            r_dimension_lower,
                            reduce_tile_width,
                            is_spatial_dynamic,
                            is_reduce_dynamic,
                            s_weight,
                            r_weight,
                            sampling_prob,
                            kMaxSamplingTimes,
                            kRepeats,
                            best_tile_config_map,
                            iter_space_type);
    }  // end of r_dimension_lower loop
  }    // end of s_dimension_lower loop
  if (test_single_large) {
    // (II) Test in the single large areas,
    // i.e., S:[4096-32768]*R:[2-1024], S:[2-1024]*R:[4096-32768]
    for (int s_dimension_lower = 2; s_dimension_lower < 1024;
         s_dimension_lower += spatial_tile_config) {
      // adjust the tile size for the spatial dimension dynamically
      spatial_tile_config =
          get_tile_size_config_in_large_area(s_dimension_lower);
      spatial_tile_width = (is_spatial_dynamic ? spatial_tile_config : 1);

      for (int r_dimension_lower = 4096; r_dimension_lower < 32768;
           r_dimension_lower += reduce_tile_config) {
        // adjust the tile size for the reduce dimension dymaically
        reduce_tile_config =
            get_tile_size_config_in_large_area(r_dimension_lower);
        reduce_tile_width = (is_reduce_dynamic ? reduce_tile_config : 1);

        TestWindowPerformance(os,
                              perf_test_config,
                              kGraphNum,
                              s_dimension_lower,
                              spatial_tile_width,
                              r_dimension_lower,
                              reduce_tile_width,
                              is_spatial_dynamic,
                              is_reduce_dynamic,
                              s_weight,
                              r_weight,
                              sampling_prob,
                              kMaxSamplingTimes,
                              kRepeats,
                              best_tile_config_map,
                              iter_space_type);
      }
    }

    for (int s_dimension_lower = 4096; s_dimension_lower < 32768;
         s_dimension_lower += spatial_tile_config) {
      // adjust the tile size for the spatial dimension dymaically
      spatial_tile_config =
          get_tile_size_config_in_large_area(s_dimension_lower);
      spatial_tile_width = (is_spatial_dynamic ? spatial_tile_config : 1);

      for (int r_dimension_lower = 2; r_dimension_lower < 1024;
           r_dimension_lower += reduce_tile_config) {
        // adjust the tile size for the reduce dimension dymaically
        reduce_tile_config =
            get_tile_size_config_in_large_area(r_dimension_lower);
        reduce_tile_width = (is_reduce_dynamic ? reduce_tile_config : 1);
        // Run performance test and write measure results to csv file.

        TestWindowPerformance(os,
                              perf_test_config,
                              kGraphNum,
                              s_dimension_lower,
                              spatial_tile_width,
                              r_dimension_lower,
                              reduce_tile_width,
                              is_spatial_dynamic,
                              is_reduce_dynamic,
                              s_weight,
                              r_weight,
                              sampling_prob,
                              kMaxSamplingTimes,
                              kRepeats,
                              best_tile_config_map,
                              iter_space_type);
      }
    }
  }
  os.close();
}

/**
 * @brief Test case for the ConfigSearcher.
 *
 * This test case performs a perormance test for the best configuration derived
 * from the ConfigSearcher. It iterates over different spatial and reduce tile
 * sizes and constructs a pir::Program. The objective function used for the
 * search is a WeightedSamplingTrailObjectiveFunc. The performance results are
 * written into a CSV file, including the default score and the best candidate
 * score.
 */

TEST(ConfigSearcher, TestPerfDynamicDynamic) {
  constexpr int spatial_left_bound = 2;   // for full test, set it to 2
  constexpr int spatial_right_bound = 2;  // for full test, set it to 4096
  constexpr int reduce_left_bound = 2;    // for full test, set it to 2
  constexpr int reduce_right_bound = 2;   // for full test, set it to 4096
  constexpr bool is_spatial_dynamic = true;
  constexpr bool is_reduce_dynamic = true;
  bool test_single_large = false;  // To test single large area, set it to true
  TestPerformanceForTileConfig(spatial_left_bound,
                               spatial_right_bound,
                               reduce_left_bound,
                               reduce_right_bound,
                               is_spatial_dynamic,
                               is_reduce_dynamic,
                               test_single_large);
}

TEST(ConfigSearcher, TestPerfStaticDynamic) {
  constexpr int spatial_left_bound = 2;   // for full test, set it to 2
  constexpr int spatial_right_bound = 2;  // for full test, set it to 4096
  constexpr int reduce_left_bound = 2;    // for full test, set it to 2
  constexpr int reduce_right_bound = 2;   // for full test, set it to 4096
  constexpr bool is_spatial_dynamic = false;
  constexpr bool is_reduce_dynamic = true;
  bool test_single_large = false;  // To test single large area, set it to true
  TestPerformanceForTileConfig(spatial_left_bound,
                               spatial_right_bound,
                               reduce_left_bound,
                               reduce_right_bound,
                               is_spatial_dynamic,
                               is_reduce_dynamic,
                               test_single_large);
}

TEST(ConfigSearcher, TestPerfDynamicStatic) {
  constexpr int spatial_left_bound = 2;   // for full test, set it to 2
  constexpr int spatial_right_bound = 2;  // for full test, set it to 4096
  constexpr int reduce_left_bound = 2;    // for full test, set it to 2
  constexpr int reduce_right_bound = 2;   // for full test, set it to 4096
  constexpr bool is_spatial_dynamic = true;
  constexpr bool is_reduce_dynamic = false;
  bool test_single_large = false;  // To test single large area, set it to true
  TestPerformanceForTileConfig(spatial_left_bound,
                               spatial_right_bound,
                               reduce_left_bound,
                               reduce_right_bound,
                               is_spatial_dynamic,
                               is_reduce_dynamic,
                               test_single_large);
}
