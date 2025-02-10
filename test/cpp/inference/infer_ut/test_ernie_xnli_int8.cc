// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "test_helper.h"  // NOLINT
#include "test_suite.h"   // NOLINT

DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_string(datadir, "", "dataset.");
DEFINE_string(truth_data, "", "Directory of the inference data truth result");

namespace paddle_infer {

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  config.SetModel(FLAGS_modeldir + "/__model__",
                  FLAGS_modeldir + "/__params__");
  config.EnableUseGpu(1000, 0);
  // Open the memory optim.
  config.EnableMemoryOptim();

  int max_batch = 32;
  int max_single_seq_len = 128;
  int opt_single_seq_len = 64;
  int min_batch_seq_len = 1;
  int max_batch_seq_len = 512;
  int opt_batch_seq_len = 256;

  std::string input_name0 = "eval_placeholder_0";
  std::string input_name1 = "eval_placeholder_1";
  std::string input_name2 = "eval_placeholder_2";
  std::string input_name3 = "eval_placeholder_3";

  std::vector<int> min_shape = {min_batch_seq_len};
  std::vector<int> max_shape = {max_batch_seq_len};
  std::vector<int> opt_shape = {opt_batch_seq_len};
  // Set the input's min, max, opt shape
  std::map<std::string, std::vector<int>> min_input_shape = {
      {input_name0, min_shape},
      {input_name1, min_shape},
      {input_name2, {1}},
      {input_name3, {1, min_batch_seq_len, 1}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {input_name0, max_shape},
      {input_name1, max_shape},
      {input_name2, {max_batch + 1}},
      {input_name3, {1, max_single_seq_len, 1}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {input_name0, opt_shape},
      {input_name1, opt_shape},
      {input_name2, {max_batch + 1}},
      {input_name3, {1, opt_single_seq_len, 1}}};

  // only kHalf supported
  config.EnableTensorRtEngine(
      1 << 30, 1, 5, Config::Precision::kInt8, false, false);
  // ernie varlen must be used with dynamic shape
  config.SetTRTDynamicShapeInfo(
      min_input_shape, max_input_shape, opt_input_shape);
  // ernie varlen must be used with oss
  config.EnableVarseqlen();
  paddle_infer::experimental::InternalUtils::SetTransformerPosid(&config,
                                                                 input_name2);
  paddle_infer::experimental::InternalUtils::SetTransformerMaskid(&config,
                                                                  input_name3);

  return CreatePredictor(config);
}

// Parse tensor from string
template <typename T>
std::vector<T> ParseTensor(const std::string &field) {
  std::string mat_str = field;

  std::vector<T> mat;
  paddle::test::Split(mat_str, ' ', &mat);

  return mat;
}

void run(Predictor *predictor, std::vector<float> *out_data) {
  clock_t start, end;
  start = clock();
  CHECK(predictor->Run());
  end = clock();

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  out_data->resize(out_num);
  output_t->CopyToCpu(out_data->data());
  return;
}

auto PrepareOutput(std::string input_file) -> std::deque<float> {
  std::ifstream fin(input_file);
  std::string line;
  std::vector<std::string> buffer;
  while (std::getline(fin, line)) {
    buffer.emplace_back(line);
  }
  std::deque<float> resDeque(buffer.size());
  std::transform(buffer.begin(),
                 buffer.end(),
                 resDeque.begin(),
                 [](const std::string &val) { return std::stof(val); });

  return resDeque;
}  // PrepareOutput

TEST(tensorrt_tester_ernie_xnli, oss_varlen_truth_data_int8) {
  auto resDeque = PrepareOutput(FLAGS_truth_data);
  auto predictor = InitPredictor();

  ASSERT_FALSE(FLAGS_datadir.empty());
  std::ifstream fin(FLAGS_datadir);
  std::string line;

  int lineno = 0;
  const int max_seq_len = 128;
  const int run_batch = 1;
  int correct_num = 0;
  while (std::getline(fin, line)) {
    std::vector<std::string> fields;
    paddle::test::Split(line, ';', &fields);

    auto src_ids = ParseTensor<int32_t>(fields[0]);
    auto sent_ids = ParseTensor<int32_t>(fields[1]);
    auto pos_ids = ParseTensor<int64_t>(fields[2]);

    int run_seq_len = src_ids.size();
    int32_t i3[2] = {0, run_seq_len};
    int32_t i4[max_seq_len] = {0};

    auto input_names = predictor->GetInputNames();

    // first input
    auto input_t1 = predictor->GetInputHandle(input_names[0]);
    input_t1->Reshape({run_seq_len});
    input_t1->CopyFromCpu(src_ids.data());

    // second input
    auto input_t2 = predictor->GetInputHandle(input_names[1]);
    input_t2->Reshape({run_seq_len});
    input_t2->CopyFromCpu(sent_ids.data());

    // third input
    auto input_t3 = predictor->GetInputHandle(input_names[2]);
    input_t3->Reshape({run_batch + 1});
    input_t3->CopyFromCpu(i3);

    // fourth input
    auto input_t4 = predictor->GetInputHandle(input_names[3]);
    input_t4->Reshape({1, max_seq_len, 1});
    input_t4->CopyFromCpu(i4);

    std::vector<float> out_data;
    run(predictor.get(), &out_data);

    lineno++;
    int maxPosition =
        max_element(out_data.begin(), out_data.end()) - out_data.begin();

    if (maxPosition == resDeque[0]) {
      correct_num += 1;
    }
    resDeque.pop_front();

    VLOG(2) << "predict result: " << maxPosition;
    for (auto r : out_data) {
      VLOG(2) << r;
    }
  }
  ASSERT_GT(correct_num,
            3855);  // total input 5010, int8 res should greater than 3855
  LOG(INFO) << "=== finish oss test ===";
}

}  // namespace paddle_infer

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if IS_TRT_VERSION_GE(7200)
  return RUN_ALL_TESTS();
#endif
  return 0;
}
