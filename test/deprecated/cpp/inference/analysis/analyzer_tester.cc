// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include <array>

#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/phi/common/port.h"

namespace paddle {
namespace inference {
namespace analysis {

using namespace framework;  // NOLINT

TEST(Analyzer, analysis_without_tensorrt) {
  Argument argument;
  argument.SetDisableLogs(false);
  argument.SetModelDir(FLAGS_inference_model_dir);
  argument.SetEnableIrOptim(false);
  argument.SetUseGPU(false);
  argument.SetUsePIR(false);
  argument.SetAnalysisPasses({"ir_graph_build_pass",
                              "ir_analysis_pass",
                              "ir_params_sync_among_devices_pass"});

  Analyzer analyser;
  analyser.Run(&argument);
}

TEST(Analyzer, analysis_with_tensorrt) {
  Argument argument;
  argument.SetDisableLogs(false);
  argument.SetEnableIrOptim(false);
  argument.SetTensorRtMaxBatchSize(3);
  argument.SetTensorRtWorkspaceSize(1 << 20);
  argument.SetModelDir(FLAGS_inference_model_dir);
  argument.SetUseGPU(false);
  argument.SetUsePIR(false);
  argument.SetAnalysisPasses({"ir_graph_build_pass",
                              "ir_analysis_pass",
                              "ir_params_sync_among_devices_pass"});

  Analyzer analyser;
  analyser.Run(&argument);
}

void TestWord2vecPrediction(const std::string& model_path) {
  NativeConfig config;
  config.model_dir = model_path;
  config.use_gpu = false;
  config.device = 0;
  auto predictor = ::paddle::CreatePaddlePredictor<NativeConfig>(config);

  // One single batch

  std::array<int64_t, 4> data = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data = PaddleBuf(data.data(), sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  // For simplicity, we set all the slots with the same data.
  std::vector<PaddleTensor> slots(4, tensor);
  std::vector<PaddleTensor> outputs;
  PADDLE_ENFORCE_EQ(
      predictor->Run(slots, &outputs),
      true,
      common::errors::Fatal("Paddle predictor failed running, please check"));

  PADDLE_ENFORCE_EQ(outputs.size(),
                    1UL,
                    common::errors::PreconditionNotMet(
                        "Output size should be 1, but got %d", outputs.size()));
  // Check the output buffer size and result of each tid.
  PADDLE_ENFORCE_EQ(outputs.front().data.length(),
                    33168UL,
                    common::errors::PreconditionNotMet(
                        "Output's data length should be 33168 but got %d",
                        outputs.front().data.length()));
  std::array<float, 5> result = {
      0.00129761, 0.00151112, 0.000423564, 0.00108815, 0.000932706};
  const size_t num_elements = outputs.front().data.length() / sizeof(float);
  // The outputs' buffers are in CPU memory.
  for (size_t i = 0; i < std::min(static_cast<size_t>(5UL), num_elements);
       i++) {
    LOG(INFO) << "data: " << static_cast<float*>(outputs.front().data.data())[i]
              << " result: " << result[i];
    EXPECT_NEAR(
        static_cast<float*>(outputs.front().data.data())[i], result[i], 1e-3);
  }
}

TEST(Analyzer, word2vec_without_analysis) {
  TestWord2vecPrediction(FLAGS_inference_model_dir);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
