/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file contains demo of mobilenet for onnxruntime backend.
 */
#include <glog/logging.h>  // use glog instead of CHECK to avoid importing other paddle header files.

#include <algorithm>
#include <numeric>
#include <vector>

#include "gflags/gflags.h"
#include "utils.h"  // NOLINT

DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_string(data, "", "path of data");

namespace paddle {
namespace demo {

/*
 * Use the onnxruntime engine to inference the demo.
 */
void Main() {
  paddle::AnalysisConfig config;
  config.EnableONNXRuntime();
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  auto predictor = paddle_infer::CreatePredictor(config);

  // Inference.
  LOG(INFO) << "--- prepare input data ----";
  std::vector<int> input_shape = {1, 3, 224, 224};
  std::vector<float> input_data;
  std::string line;
  std::ifstream file(FLAGS_data);
  std::getline(file, line);
  file.close();
  std::vector<std::string> data_strs;
  split(line, ' ', &data_strs);
  int input_num = 0;
  for (auto& d : data_strs) {
    input_num += 1;
    input_data.push_back(std::stof(d));
  }

  std::vector<float> out_data;
  out_data.resize(1000);
  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto input_tensor = predictor->GetInputHandle(input_names[0]);
  input_tensor->Reshape(input_shape);
  auto output_tensor = predictor->GetOutputHandle(output_names[0]);

  input_tensor->CopyFromCpu(input_data.data());
  predictor->Run();
  output_tensor->CopyToCpu(out_data.data());

  std::vector<int> out_index(out_data.size());
  std::iota(out_index.begin(), out_index.end(), 0);
  std::sort(
      out_index.begin(), out_index.end(), [&out_data](int index1, int index2) {
        return out_data[index1] > out_data[index2];
      });
  LOG(INFO) << "output.size " << out_data.size()
            << "  max_index:" << out_index[0];
  PADDLE_ENFORCE_EQ(out_data.size(),
                    1000,
                    common::errors::InvalidArgument(
                        "Required out_data.size() should be equal to 1000. "));
  int max_index = out_index[0];
  PADDLE_ENFORCE_EQ(max_index,
                    13,
                    common::errors::InvalidArgument(
                        "Required max_index should be equal to 13. "));
  float max_score = out_data[max_index];
  PADDLE_ENFORCE_LE(fabs(max_score - 0.99981),
                    1e-4,
                    common::errors::InvalidArgument(
                        "Required fabs(max_score - 0.99981) should "
                        "be less than or equal to 1e-4. "));
}

}  // namespace demo
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  paddle::demo::Main();
  return 0;
}
