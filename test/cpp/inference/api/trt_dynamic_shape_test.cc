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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/common/flags.h"
#include "test/cpp/inference/api/trt_test_helper.h"

namespace paddle {
namespace inference {

void TestDynamic(bool with_dynamic = true,
                 bool delete_cache = true,
                 bool delete_conv_bn = false) {
  std::string model_dir =
      FLAGS_infer_model + "/conv_bn_swish_split_gelu/conv_bn_swish_split_gelu";

  std::string opt_cache_dir = model_dir + "/my_cache";
  if (delete_cache) {
    delete_cache_files(opt_cache_dir);
  }

  AnalysisConfig config;
  config.EnableNewIR(false);
  config.EnableUseGpu(100, 0);
  std::string buffer_prog, buffer_param;
  ReadBinaryFile(model_dir + "/model", &buffer_prog);
  ReadBinaryFile(model_dir + "/params", &buffer_param);
  config.SetModelBuffer(&buffer_prog[0],
                        buffer_prog.size(),
                        &buffer_param[0],
                        buffer_param.size());
  config.SetOptimCacheDir(opt_cache_dir);

  // Set the input's min, max, opt shape
  config.EnableTensorRtEngine(
      1 << 30, 1, 1, AnalysisConfig::Precision::kFloat32, true, true);
  if (delete_conv_bn) {
    config.pass_builder()->DeletePass("conv_bn_fuse_pass");
  }
  if (with_dynamic) {
    std::map<std::string, std::vector<int>> min_input_shape = {
        {"image", {1, 1, 3, 3}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"image", {1, 1, 10, 10}}};
    std::map<std::string, std::vector<int>> opt_input_shape = {
        {"image", {1, 1, 3, 3}}};

    config.SetTRTDynamicShapeInfo(
        min_input_shape, max_input_shape, opt_input_shape);
  }
  auto predictor = CreatePaddlePredictor(config);
  auto input_names = predictor->GetInputNames();
  int channels = 1;
  int height = 3;
  int width = 3;
  int input_num = channels * height * width * 1;

  float *input = new float[input_num];
  memset(input, 0, input_num * sizeof(float));
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({1, channels, height, width});
  input_t->copy_from_cpu(input);

  ASSERT_TRUE(predictor->ZeroCopyRun());

  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());
}

void TestDynamic2() {
  std::string model_dir =
      FLAGS_infer_model + "/complex_model_dynamic/complex_model_dynamic2";
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  // Set the input's min, max, opt shape
  int batch_size = 1;
  std::map<std::string, std::vector<int>> min_input_shape = {
      {"image", {1, 3, 3, 3}}, {"in1", {1, 2, 1, 1}}, {"in2", {1, 2, 1, 1}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {"image", {1, 3, 10, 10}}, {"in1", {1, 2, 1, 1}}, {"in2", {1, 2, 1, 1}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {"image", {1, 3, 5, 5}}, {"in1", {1, 2, 1, 1}}, {"in2", {1, 2, 1, 1}}};
  config.EnableTensorRtEngine(
      1 << 30, batch_size, 0, AnalysisConfig::Precision::kFloat32, false, true);

  config.SetTRTDynamicShapeInfo(
      min_input_shape, max_input_shape, opt_input_shape);

  auto predictor = CreatePaddlePredictor(config);
  int channels = 3;
  int height = 5;
  int width = 5;
  int input_num = channels * height * width * 1;

  float *input = new float[input_num];
  memset(input, 0, input_num * sizeof(float));
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->copy_from_cpu(input);

  auto input_t1 = predictor->GetInputTensor(input_names[1]);
  input_t1->Reshape({batch_size, 2, 1, 1});
  std::vector<float> first;
  for (int i = 0; i < batch_size * 2; i++) first.push_back(1.0);
  input_t1->copy_from_cpu(first.data());

  auto input_t2 = predictor->GetInputTensor(input_names[2]);
  input_t2->Reshape({batch_size, 2, 1, 1});
  input_t2->copy_from_cpu(first.data());

  ASSERT_TRUE(predictor->ZeroCopyRun());

  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());
  std::vector<float> result = {0.617728, 1.63504, 2.15771, 0.535556};
  for (size_t i = 0; i < out_data.size(); i++) {
    EXPECT_NEAR(result[i], out_data[i], 1e-5);
  }
}

void TestTunedDynamic() {
  std::string model_dir =
      FLAGS_infer_model + "/complex_model_dynamic/complex_model_dynamic2";
  AnalysisConfig config_tuned;
  const std::string shape_range = "shape_range.pbtxt";
  config_tuned.EnableUseGpu(100, 0);
  config_tuned.SetModel(model_dir + "/model", model_dir + "/params");
  config_tuned.CollectShapeRangeInfo(shape_range);

  int batch_size = 1;
  auto predictor_tuned = CreatePaddlePredictor(config_tuned);

  auto check_func = [batch_size](PaddlePredictor *predictor) {
    int channels = 3;
    int height = 5;
    int width = 5;
    int input_num = channels * height * width * 1;

    float *input = new float[input_num];
    memset(input, 0, input_num * sizeof(float));
    auto input_names = predictor->GetInputNames();
    auto input_t = predictor->GetInputTensor(input_names[0]);
    input_t->Reshape({batch_size, channels, height, width});
    input_t->copy_from_cpu(input);

    auto input_t1 = predictor->GetInputTensor(input_names[1]);
    input_t1->Reshape({batch_size, 2, 1, 1});
    std::vector<float> first;
    for (int i = 0; i < batch_size * 2; i++) first.push_back(1.0);
    input_t1->copy_from_cpu(first.data());

    auto input_t2 = predictor->GetInputTensor(input_names[2]);
    input_t2->Reshape({batch_size, 2, 1, 1});
    input_t2->copy_from_cpu(first.data());

    ASSERT_TRUE(predictor->ZeroCopyRun());

    std::vector<float> out_data;
    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputTensor(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(
        output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data.resize(out_num);
    output_t->copy_to_cpu(out_data.data());
  };
  check_func(predictor_tuned.get());
  predictor_tuned.reset(nullptr);

  // check tuned_dynamic_shape
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  std::string cache_dir = "tuned_cache";
  config.SetOptimCacheDir(cache_dir);
  delete_cache_files(cache_dir);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableTunedTensorRtDynamicShape(shape_range, true);
  config.EnableTensorRtEngine(
      1 << 30, batch_size, 0, AnalysisConfig::Precision::kFloat32, true, false);
  auto test_predictor = CreatePaddlePredictor(config);
  check_func(test_predictor.get());
}

void TestDynamicClone(bool with_dynamic = true,
                      bool delete_cache = true,
                      bool delete_conv_bn = false) {
  std::string model_dir =
      FLAGS_infer_model + "/conv_bn_swish_split_gelu/conv_bn_swish_split_gelu";

  std::string opt_cache_dir = model_dir + "/my_cache";
  if (delete_cache) {
    delete_cache_files(opt_cache_dir);
  }

  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  std::string buffer_prog, buffer_param;
  ReadBinaryFile(model_dir + "/model", &buffer_prog);
  ReadBinaryFile(model_dir + "/params", &buffer_param);
  config.SetModelBuffer(&buffer_prog[0],
                        buffer_prog.size(),
                        &buffer_param[0],
                        buffer_param.size());
  config.SetOptimCacheDir(opt_cache_dir);

  // Set the input's min, max, opt shape
  config.EnableTensorRtEngine(
      1 << 30, 1, 1, AnalysisConfig::Precision::kFloat32, false, false);
  if (delete_conv_bn) {
    config.pass_builder()->DeletePass("conv_bn_fuse_pass");
  }
  if (with_dynamic) {
    std::map<std::string, std::vector<int>> min_input_shape = {
        {"image", {1, 1, 3, 3}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"image", {1, 1, 10, 10}}};
    std::map<std::string, std::vector<int>> opt_input_shape = {
        {"image", {1, 1, 3, 3}}};

    config.SetTRTDynamicShapeInfo(
        min_input_shape, max_input_shape, opt_input_shape);
  }
  auto predictor = CreatePaddlePredictor(config);
  auto input_names = predictor->GetInputNames();
  int channels = 1;
  int height = 3;
  int width = 3;
  int input_num = channels * height * width * 1;

  float *input = new float[input_num];
  memset(input, 0, input_num * sizeof(float));
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({1, channels, height, width});
  input_t->copy_from_cpu(input);

  ASSERT_TRUE(predictor->ZeroCopyRun());

  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());

  auto predictor2 = predictor->Clone();
  auto input_t2 = predictor2->GetInputTensor(input_names[0]);
  input_t2->Reshape({1, channels, height, width});
  input_t2->copy_from_cpu(input);

  ASSERT_TRUE(predictor2->ZeroCopyRun());

  std::vector<float> out_data2;
  auto output_t2 = predictor2->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape2 = output_t2->shape();
  int out_num2 = std::accumulate(
      output_shape2.begin(), output_shape2.end(), 1, std::multiplies<int>());
  out_data2.resize(out_num2);
  output_t2->copy_to_cpu(out_data2.data());
  ASSERT_TRUE(out_data2.size() == out_data.size());
  for (size_t i = 0; i < out_data.size(); i++) {
    EXPECT_NEAR(out_data2[i], out_data[i], 1e-5);
  }
}

TEST(AnalysisPredictor, trt_dynamic) { TestDynamic(true); }
TEST(AnalysisPredictor, trt_memory_serialize) {
  // serialize
  TestDynamic(true, true, true);
  // deserialize
  TestDynamic(true, false, true);
}
TEST(AnalysisPredictor, trt_dynamic2) { TestDynamic2(); }

TEST(AnalysisPredictor, trt_tuned_dynamic) { TestTunedDynamic(); }
TEST(AnalysisPredictor, trt_dynamic_clone) { TestDynamicClone(); }

}  // namespace inference
}  // namespace paddle
