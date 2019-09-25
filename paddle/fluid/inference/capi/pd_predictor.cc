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

#include <algorithm>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"

extern "C" {

bool PD_PredictorRun(PD_Predictor* predictor, const PD_Tensor* inputs,
                     PD_Tensor* output_data, int batch_size = -1) {
  return predictor->predictor->Run(inputs, output_data, batch_size);
}

char** PD_GetPredictorInputNames(PD_Predictor* predictor) {
  std::vector<std::string> ret_names;
  ret_names = predictor->predictor->GetInputNames();
  int size = ret_names.size();
  char** names;
  for (int i = 0; i < size; ++i) {
    names[i] = ret_names[i].data();
  }
  return names;
}

InTensorShape* PD_GetPredictorInputTensorShape(PD_Predictor* predictor,
                                               int* size) {
  std::unordered_map<std::string, std::vector<int64_t>> input_tensor_shape =
      predictor->predictor->GetInputTensorShape();
  InTensorShape* ret_in_tensor_shape;
  int i = 0;
  for (auto item : input_tensor_shape) {
    ret_in_tensor_shape[i].name = item.first.data();
    std::vector<int64_t> tmp_shape = item.second;
    ret_in_tensor_shape[i].shape_size = tmp_shape.size();
    for (int j = 0; j < tmp_shape.size(); ++j) {
      ret_in_tensor_shape[i].tensor_shape[j] = tmp_shape[j];
    }
    ++i;
  }
  size = &i;
  return ret_in_tensor_shape;
}

char** PD_GetPredictorOutputNames(PD_Predictor* predictor) {
  std::vector<std::string> ret_names;
  ret_names = predictor->predictor->GetOutputNames();
  int size = ret_names.size();
  char** names;
  for (int i = 0; i < size; ++i) {
    names[i] = ret_names[i].data();
  }
  return names;
}

PD_ZeroCopyTensor* PD_GetPredictorInputTensor(PD_Predictor* predictor,
                                              const char* name) {
  return predictor->predictor->GetInputTensor(std::string(name)).get();
}

PD_ZeroCopyTensor* PD_GetPredictorOutputTensor(PD_Predictor* predictor,
                                               const char* name) {
  return predictor->predictor->GetOutputTensor(std::string(name)).get();
}

bool PD_PredictorZeroCopyRun(PD_Predictor* predictor) {
  return predictor->predictor->ZeroCopyRun();
}

void* PD_DeletePredictor(PD_Predictor* predictor) {
  if (predictor) {
    delete predictor;
    predictor = nullptr;
  }
}

PD_Predictor* PD_ClonePredictor(const PD_Predictor* predictor) {
  PD_Predictor* cloned = new PD_Predictor;
  cloned->predictor = predictor->predictor->Clone();
  return cloned;
}

PD_Predictor* PD_NewPredictor(const PD_AnalysisConfig* config) {
  auto predictor = new PD_Predictor();
  predictor->predictor = paddle::CreatePaddlePredictor(config->config);

  return predictor;
}
}  // extern "C"
