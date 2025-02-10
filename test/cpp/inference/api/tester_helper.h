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

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef WITH_GPERFTOOLS
#include <gperftools/profiler.h>
#endif
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "test/cpp/inference/api/config_printer.h"
#include "test/cpp/inference/test_helper.h"

PD_DEFINE_string(model_name, "", "model name");
PD_DEFINE_string(infer_model, "", "model path");
PD_DEFINE_string(fp32_model, "", "FP32 model path");
PD_DEFINE_string(int8_model, "", "INT8 model path");
PD_DEFINE_string(infer_data, "", "data file");
PD_DEFINE_string(refer_result, "", "reference result for comparison");
PD_DEFINE_int32(batch_size, 1, "batch size");
PD_DEFINE_bool(ernie_large, false, "Test ernie large");
PD_DEFINE_bool(with_accuracy_layer,
               true,
               "Calculate the accuracy while label is in the input");
PD_DEFINE_bool(enable_fp32, true, "Enable FP32 type prediction");
PD_DEFINE_bool(enable_bf16, false, "Enable BF16 type prediction");
PD_DEFINE_bool(enable_int8_ptq,
               false,
               "Enable INT8 post-training quantization prediction");
PD_DEFINE_bool(enable_int8_qat,
               false,
               "Enable INT8 quant-aware training prediction");
PD_DEFINE_int32(warmup_batch_size, 100, "batch size for quantization warmup");
// setting iterations to 0 means processing the whole dataset
PD_DEFINE_int32(iterations, 0, "number of batches to process");
PD_DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
PD_DEFINE_bool(test_all_data, false, "Test the all dataset in data file.");
PD_DEFINE_int32(num_threads,
                1,
                "Running the inference program in multi-threads.");
PD_DEFINE_bool(use_analysis,
               true,
               "Running the inference program in analysis mode.");
PD_DEFINE_double(accuracy, 1e-3, "Result Accuracy.");
PD_DEFINE_double(quantized_accuracy, 2e-2, "Result Quantized Accuracy.");
PD_DEFINE_bool(zero_copy, false, "Use ZeroCopy to speedup Feed/Fetch.");
PD_DEFINE_bool(warmup,
               false,
               "Use warmup to calculate elapsed_time more accurately. "
               "To reduce CI time, it sets false in default.");
PD_DEFINE_int32(warmup_iters, 1, "Number of batches to process during warmup.");

PD_DEFINE_bool(enable_profile, false, "Turn on profiler for fluid");
PD_DEFINE_int32(cpu_num_threads,
                1,
                "Number of threads for each paddle instance.");
PD_DEFINE_bool(fuse_multi_gru,
               false,
               "Running the inference program with multi_gru_fuse_pass");

// ipu related
PD_DEFINE_int32(ipu_micro_batch_size, 1, "micro batch size");
PD_DEFINE_int32(ipu_device_num, 1, "device num");
PD_DEFINE_bool(ipu_enable_pipelining, false, "enable pipelining");
PD_DEFINE_int32(ipu_batches_per_step,
                1,
                "the number of batches per run in pipelining");
PD_DEFINE_bool(ipu_enable_fp16, false, "enable fp16");
PD_DEFINE_int32(ipu_replica_num, 1, "replica num");
PD_DEFINE_double(ipu_available_memory_proportion,
                 1.0,
                 "available memory proportion");
PD_DEFINE_bool(ipu_enable_half_partial, false, "enable half partial");

namespace paddle {
namespace inference {

using ::paddle::framework::proto::VarType;
using float16 = ::phi::dtype::float16;

template <typename T>
constexpr ::paddle::PaddleDType GetPaddleDType();

template <>
constexpr ::paddle::PaddleDType GetPaddleDType<int64_t>() {
  return ::paddle::PaddleDType::INT64;
}

template <>
constexpr ::paddle::PaddleDType GetPaddleDType<float>() {
  return ::paddle::PaddleDType::FLOAT32;
}

void PrintConfig(const PaddlePredictor::Config *config, bool use_analysis) {
  const auto *analysis_config =
      reinterpret_cast<const AnalysisConfig *>(config);
  if (use_analysis) {
    LOG(INFO) << *analysis_config;
    return;
  }
  LOG(INFO) << analysis_config->ToNativeConfig();
}

void CheckError(float data_ref, float data) {
  if (std::abs(data_ref) > 1) {
    PADDLE_ENFORCE_LE(
        std::abs((data_ref - data) / data_ref),
        FLAGS_accuracy,
        common::errors::InvalidArgument(
            "[Error info] abs((data_ref - data) / data_ref) must be less than "
            "or equal to FLAGS_accuracy.\n"
            "[Argument info] Please check your input data_ref and data."));
  } else {
    PADDLE_ENFORCE_LE(
        std::abs(data_ref - data),
        FLAGS_accuracy,
        common::errors::InvalidArgument(
            "[Error info] abs(data_ref - data) must be less than or equal to "
            "FLAGS_accuracy.\n"
            "[Argument info] Please check your input data_ref and data."));
  }
}

class Barrier {
 public:
  explicit Barrier(std::size_t count) : _count(count) {}
  void Wait() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (--_count) {
      _cv.wait(lock, [this] { return _count == 0; });
    } else {
      _cv.notify_all();
    }
  }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::size_t _count;
};

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file,
               size_t beginning_offset,
               std::vector<int> shape,
               std::string name)
      : file_(file), position_(beginning_offset), shape_(shape), name_(name) {
    numel_ = std::accumulate(
        shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
  }

  PaddleTensor NextBatch() {
    PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape_;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel_ * sizeof(T));

    file_.seekg(position_);
    file_.read(static_cast<char *>(tensor.data.data()), numel_ * sizeof(T));
    position_ = file_.tellg();

    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");

    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position_;
  std::vector<int> shape_;
  std::string name_;
  size_t numel_;
};

std::shared_ptr<std::vector<PaddleTensor>> GetWarmupData(
    const std::vector<std::vector<PaddleTensor>> &test_data,
    int num_images = FLAGS_warmup_batch_size) {
  int test_data_batch_size = test_data[0][0].shape[0];
  auto iterations = test_data.size();
  auto all_test_data_size = iterations * test_data_batch_size;
  PADDLE_ENFORCE_LE(static_cast<size_t>(num_images),
                    all_test_data_size,
                    common::errors::InvalidArgument(
                        "The requested quantization warmup data size must be "
                        "lower or equal to the test data size. But received"
                        "warmup size is %d and test data size is %d. Please "
                        "use --warmup_batch_size parameter to set smaller "
                        "warmup batch size.",
                        num_images,
                        all_test_data_size));

  PaddleTensor images;
  images.name = "image";
  images.shape = {num_images, 3, 224, 224};
  images.dtype = PaddleDType::FLOAT32;
  images.data.Resize(sizeof(float) * num_images * 3 * 224 * 224);

  PaddleTensor labels;
  labels.name = "label";
  labels.shape = {num_images, 1};
  labels.dtype = PaddleDType::INT64;
  labels.data.Resize(sizeof(int64_t) * num_images);

  for (int i = 0; i < num_images; i++) {
    auto batch = i / test_data_batch_size;
    auto element_in_batch = i % test_data_batch_size;
    std::copy_n(static_cast<float *>(test_data[batch][0].data.data()) +
                    element_in_batch * 3 * 224 * 224,
                3 * 224 * 224,
                static_cast<float *>(images.data.data()) + i * 3 * 224 * 224);
    if (FLAGS_with_accuracy_layer)
      std::copy_n(static_cast<int64_t *>(test_data[batch][1].data.data()) +
                      element_in_batch,
                  1,
                  static_cast<int64_t *>(labels.data.data()) + i);
  }
  auto warmup_data = std::make_shared<std::vector<PaddleTensor>>(
      FLAGS_with_accuracy_layer ? 2 : 1);
  (*warmup_data)[0] = std::move(images);
  if (FLAGS_with_accuracy_layer) (*warmup_data)[1] = std::move(labels);
  return warmup_data;
}

void SetInputs(std::vector<std::vector<PaddleTensor>> *inputs,
               int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    FAIL() << "Couldn't open file: " << FLAGS_infer_data;
  }

  int64_t total_images{0};
  file.read(reinterpret_cast<char *>(&total_images), sizeof(total_images));
  LOG(INFO) << "Total images in file: " << total_images;

  std::vector<int> image_batch_shape{batch_size, 3, 224, 224};
  std::vector<int> label_batch_shape{batch_size, 1};
  auto images_offset_in_file = static_cast<size_t>(file.tellg());
  auto labels_offset_in_file =
      images_offset_in_file + sizeof(float) * total_images * 3 * 224 * 224;

  TensorReader<float> image_reader(
      file, images_offset_in_file, image_batch_shape, "image");
  TensorReader<int64_t> label_reader(
      file, labels_offset_in_file, label_batch_shape, "label");

  auto iterations_max = total_images / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }
  for (auto i = 0; i < iterations; i++) {
    auto images = image_reader.NextBatch();
    std::vector<PaddleTensor> tmp_vec;
    tmp_vec.push_back(std::move(images));
    if (FLAGS_with_accuracy_layer) {
      auto labels = label_reader.NextBatch();
      tmp_vec.push_back(std::move(labels));
    }
    inputs->push_back(std::move(tmp_vec));
  }
}

// Compare result between two PaddleTensor
void CompareResult(const std::vector<PaddleTensor> &outputs,
                   const std::vector<PaddleTensor> &ref_outputs) {
  EXPECT_GT(outputs.size(), 0UL);
  EXPECT_EQ(outputs.size(), ref_outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &out = outputs[i];
    auto &ref_out = ref_outputs[i];
    size_t size = VecReduceToInt(out.shape);
    size_t ref_size = VecReduceToInt(ref_out.shape);
    EXPECT_GT(size, 0UL);
    EXPECT_EQ(size, ref_size);
    EXPECT_EQ(out.dtype, ref_out.dtype);

#define COMPARE(paddle_type, type, func)                        \
  case paddle_type: {                                           \
    type *pdata = static_cast<type *>(out.data.data());         \
    type *pdata_ref = static_cast<type *>(ref_out.data.data()); \
    for (size_t j = 0; j < size; ++j) {                         \
      func(pdata_ref[j], pdata[j]);                             \
    }                                                           \
    break;                                                      \
  }

    switch (out.dtype) {
      COMPARE(PaddleDType::INT64, int64_t, EXPECT_EQ);
      COMPARE(PaddleDType::FLOAT32, float, CheckError);
      COMPARE(PaddleDType::INT32, int32_t, EXPECT_EQ);
      COMPARE(PaddleDType::UINT8, uint8_t, EXPECT_EQ);
      COMPARE(PaddleDType::INT8, int8_t, EXPECT_EQ);
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "VarMessageToVarType: Unsupported dtype %d",
            static_cast<int>(out.dtype)));
    }
#undef COMPARE
  }
}

// Compare result between a PaddleTensor and a ZeroCopyTensor
void CompareResult(const std::vector<PaddleTensor> &outputs,
                   const std::vector<ZeroCopyTensor> &ref_outputs) {
  EXPECT_GT(outputs.size(), 0UL);
  EXPECT_EQ(outputs.size(), ref_outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &out = outputs[i];
    auto &ref_out = ref_outputs[i];
    size_t size = VecReduceToInt(out.shape);
    EXPECT_GT(size, 0UL);
    int ref_size = 0;  // this is the number of elements not memory size
    PaddlePlace place;

#define COMPARE(paddle_type, type, func)                     \
  case paddle_type: {                                        \
    type *pdata = static_cast<type *>(out.data.data());      \
    type *pdata_ref = ref_out.data<type>(&place, &ref_size); \
    EXPECT_EQ(size, static_cast<size_t>(ref_size));          \
    for (size_t j = 0; j < size; ++j) {                      \
      func(pdata_ref[j], pdata[j]);                          \
    }                                                        \
    break;                                                   \
  }

    switch (out.dtype) {
      COMPARE(PaddleDType::INT64, int64_t, EXPECT_EQ);
      COMPARE(PaddleDType::FLOAT32, float, CheckError);
      COMPARE(PaddleDType::INT32, int32_t, EXPECT_EQ);
      COMPARE(PaddleDType::UINT8, uint8_t, EXPECT_EQ);
      COMPARE(PaddleDType::INT8, int8_t, EXPECT_EQ);
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "VarMessageToVarType: Unsupported dtype %d",
            static_cast<int>(out.dtype)));
    }
#undef COMPARE
  }
}

std::unique_ptr<PaddlePredictor> CreateTestPredictor(
    const PaddlePredictor::Config *config, bool use_analysis = true) {
  const auto *analysis_config =
      reinterpret_cast<const AnalysisConfig *>(config);
  if (use_analysis) {
    return CreatePaddlePredictor<AnalysisConfig>(*analysis_config);
  }
  auto native_config = analysis_config->ToNativeConfig();
  return CreatePaddlePredictor<NativeConfig>(native_config);
}

size_t GetSize(const PaddleTensor &out) { return VecReduceToInt(out.shape); }

void SetFakeImageInput(std::vector<std::vector<PaddleTensor>> *inputs,
                       const std::string &dirname,
                       bool is_combined = true,
                       std::string model_filename = "model",
                       std::string params_filename = "params",
                       const std::vector<std::string> *feed_names = nullptr,
                       const int continuous_input_index = 0) {
  // Set fake_image_data
  PADDLE_ENFORCE_EQ(FLAGS_test_all_data,
                    0,
                    common::errors::InvalidArgument(
                        "In SetFakeImageInput, expected test_all_data = false, "
                        "but now test_all_data=",
                        FLAGS_test_all_data));
  std::vector<std::vector<int64_t>> feed_target_shapes = GetFeedTargetShapes(
      dirname, is_combined, model_filename, params_filename);
  std::ostringstream os;
  for (size_t i = 0; i < feed_target_shapes.size(); ++i) {
    os << "feed target " << i << ": {" << feed_target_shapes[i][0];
    for (size_t j = 1; j < feed_target_shapes[i].size(); ++j) {
      os << ", " << feed_target_shapes[i][j];
    }
    os << "}\n";
  }
  LOG(INFO) << os.str();
  if (feed_names) {
    PADDLE_ENFORCE_EQ(
        feed_names->size(),
        feed_target_shapes.size(),
        common::errors::InvalidArgument(
            "The size of feeds_names and size of "
            "feed_target_shapes must be equal, but now feeds_names "
            "size is %d and feed_target_shapes size is %d",
            feed_names->size(),
            feed_target_shapes.size()));
  }
  std::vector<PaddleTensor> input_slots(feed_target_shapes.size());
  for (size_t i = 0; i < feed_target_shapes.size(); ++i) {
    const auto &feed_shape = feed_target_shapes[i];
    auto &input = input_slots[i];
    std::vector<int> shape({FLAGS_batch_size});
    for (size_t s = 1; s < feed_shape.size(); ++s) {
      shape.push_back(static_cast<int>(feed_shape[s]));
    }
    if (feed_names) {
      input.name = (*feed_names)[i];
    }
    input.shape = shape;
    input.dtype = PaddleDType::FLOAT32;
    size_t len = std::accumulate(
        shape.begin(), shape.end(), size_t{1}, [](int a, int b) {
          return a * b;
        });
    input.data.Resize(len * sizeof(float));
    input.lod.assign({{0, static_cast<size_t>(FLAGS_batch_size)}});
    float *input_data = static_cast<float *>(input.data.data());
    // fill input data, for profile easily, do not use random data here.
    for (size_t j = 0; j < len; ++j) {
      *(input_data + j) =
          static_cast<float>((j + continuous_input_index) % len) / len;
    }
  }
  (*inputs).emplace_back(input_slots);
}

void GetInputPerBatch(const std::vector<std::vector<int64_t>> &in,
                      std::vector<std::vector<int64_t>> *out,
                      std::vector<size_t> *lod,
                      size_t batch_iter,
                      size_t batch_end) {
  lod->clear();
  lod->push_back(0);
  for (auto it = in.begin() + batch_iter; it < in.begin() + batch_end; it++) {
    out->push_back(*it);
    lod->push_back(lod->back() + (*it).size());  // calculate lod
  }
}

void ConvertPaddleTensorToZeroCopyTensor(
    PaddlePredictor *predictor, const std::vector<PaddleTensor> &inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    auto tensor = predictor->GetInputTensor(input.name);
    tensor->Reshape(input.shape);
    tensor->SetLoD({input.lod});
    if (input.dtype == PaddleDType::INT64) {
      ZeroCopyTensorAssignData<int64_t>(tensor.get(), input.data);
    } else if (input.dtype == PaddleDType::FLOAT32) {
      ZeroCopyTensorAssignData<float>(tensor.get(), input.data);
    } else if (input.dtype == PaddleDType::INT32) {
      ZeroCopyTensorAssignData<int32_t>(tensor.get(), input.data);
    } else if (input.dtype == PaddleDType::UINT8) {
      ZeroCopyTensorAssignData<uint8_t>(tensor.get(), input.data);
    } else {
      LOG(ERROR) << "unsupported feed type " << input.dtype;
    }
  }
}

void PredictionWarmUp(PaddlePredictor *predictor,
                      const std::vector<std::vector<PaddleTensor>> &inputs,
                      std::vector<std::vector<PaddleTensor>> *outputs,
                      int num_threads,
                      int tid,
                      const VarType::Type data_type = VarType::FP32) {
  int batch_size = FLAGS_batch_size;
  LOG(INFO) << "Running thread " << tid << ", warm up run...";
  if (FLAGS_zero_copy) {
    ConvertPaddleTensorToZeroCopyTensor(predictor, inputs[0]);
  }
  int iterations = 1;
  if (FLAGS_warmup_iters > 1)
    iterations =
        (std::min)(FLAGS_warmup_iters, static_cast<int>(inputs.size()));
  outputs->resize(iterations);
  Timer warmup_timer;
  double elapsed_time = 0;
  if (!FLAGS_zero_copy) {
    for (int i = 0; i < iterations; ++i) {
      warmup_timer.tic();
      predictor->Run(inputs[i], &(*outputs)[i], batch_size);
      elapsed_time += warmup_timer.toc();
    }
  } else {
    for (int i = 0; i < iterations; ++i) {
      warmup_timer.tic();
      predictor->ZeroCopyRun();
      elapsed_time += warmup_timer.toc();
    }
  }
  auto batch_latency = elapsed_time / iterations;
  PrintTime(
      batch_size, 1, num_threads, tid, batch_latency, iterations, data_type);
  if (FLAGS_enable_profile) {
    ::paddle::platform::ResetProfiler();
  }
}

void PredictionRun(PaddlePredictor *predictor,
                   const std::vector<std::vector<PaddleTensor>> &inputs,
                   std::vector<std::vector<PaddleTensor>> *outputs,
                   int num_threads,
                   int tid,
                   const VarType::Type data_type = VarType::FP32,
                   float *sample_latency = nullptr) {
  int num_times = FLAGS_repeat;
  int iterations = inputs.size();  // process the whole dataset ...
  if (FLAGS_iterations > 0 &&
      FLAGS_iterations < static_cast<int64_t>(inputs.size()))
    iterations =
        FLAGS_iterations;  // ... unless the number of iterations is set
  outputs->resize(iterations);
  LOG(INFO) << "Thread " << tid << ", number of threads " << num_threads
            << ", run " << num_times << " times...";
  Timer run_timer;
  double elapsed_time = 0;
#ifdef WITH_GPERFTOOLS
  ProfilerStart("paddle_inference.prof");
#endif
  int predicted_num = 0;
  if (!FLAGS_zero_copy) {
    for (int i = 0; i < iterations; i++) {
      run_timer.tic();
      for (int j = 0; j < num_times; j++) {
        predictor->Run(inputs[i], &(*outputs)[i], FLAGS_batch_size);
      }
      elapsed_time += run_timer.toc();

      predicted_num += FLAGS_batch_size;
      if (predicted_num % 100 == 0) {
        LOG(INFO) << predicted_num << " samples";
      }
    }
  } else {
    for (int i = 0; i < iterations; i++) {
      ConvertPaddleTensorToZeroCopyTensor(predictor, inputs[i]);
      run_timer.tic();
      for (int j = 0; j < num_times; j++) {
        predictor->ZeroCopyRun();
      }
      elapsed_time += run_timer.toc();

      predicted_num += FLAGS_batch_size;
      if (predicted_num % 100 == 0) {
        LOG(INFO) << predicted_num << " samples";
      }
    }
  }

#ifdef WITH_GPERFTOOLS
  ProfilerStop();
#endif

  auto batch_latency = elapsed_time / (iterations * num_times);
  PrintTime(FLAGS_batch_size,
            num_times,
            num_threads,
            tid,
            batch_latency,
            iterations,
            data_type);

  if (sample_latency != nullptr)
    *sample_latency = batch_latency / FLAGS_batch_size;
}

void TestOneThreadPrediction(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<std::vector<PaddleTensor>> *outputs,
    bool use_analysis = true,
    const VarType::Type data_type = VarType::FP32,
    float *sample_latency = nullptr) {
  auto predictor = CreateTestPredictor(config, use_analysis);
  if (FLAGS_warmup) {
    PredictionWarmUp(predictor.get(), inputs, outputs, 1, 0, data_type);
  }
  PredictionRun(
      predictor.get(), inputs, outputs, 1, 0, data_type, sample_latency);
}

void TestMultiThreadPrediction(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<std::vector<PaddleTensor>> *outputs,
    int num_threads,
    bool use_analysis = true) {
  std::vector<std::thread> threads;
  std::vector<std::unique_ptr<PaddlePredictor>> predictors;
  predictors.emplace_back(CreateTestPredictor(config, use_analysis));
  for (int tid = 1; tid < num_threads; tid++) {
    predictors.emplace_back(predictors.front()->Clone());
  }

  for (int tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      // Each thread should have local inputs and outputs.
      // The inputs of each thread are all the same.
      std::vector<std::vector<PaddleTensor>> outputs_tid;
      auto &predictor = predictors[tid];
      if (FLAGS_warmup) {
        PredictionWarmUp(
            predictor.get(), inputs, &outputs_tid, num_threads, tid);
      }
      PredictionRun(predictor.get(), inputs, &outputs_tid, num_threads, tid);
    });
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
}

void TestPrediction(const PaddlePredictor::Config *config,
                    const std::vector<std::vector<PaddleTensor>> &inputs,
                    std::vector<std::vector<PaddleTensor>> *outputs,
                    int num_threads,
                    bool use_analysis = FLAGS_use_analysis) {
  PrintConfig(config, use_analysis);
  if (num_threads == 1) {
    TestOneThreadPrediction(config, inputs, outputs, use_analysis);
  } else {
    TestMultiThreadPrediction(
        config, inputs, outputs, num_threads, use_analysis);
  }
}

void SummarizeAccuracy(float avg_acc_ref, float avg_acc, int compared_idx) {
  std::string data_type_name = "INT8";
  if (FLAGS_enable_bf16) data_type_name = "BF16";
  PADDLE_ENFORCE_LE(
      compared_idx,
      2,
      common::errors::InvalidArgument(
          "The compared_idx should be <= 2. But received compared_idx = %d. "
          "For top1 accuracy, set compared_idx = 1; For top5 accuracy or mean "
          "Average Precision (mAP), set compared_idx = 2.",
          compared_idx));
  PADDLE_ENFORCE_GE(
      compared_idx,
      1,
      common::errors::InvalidArgument(
          "The compared_idx should be >= 1. But received compared_idx = %d. "
          "For top1 accuracy, set compared_idx = 1; For top5 accuracy or mean "
          "Average Precision (mAP), set compared_idx = 2.",
          compared_idx));
  std::string prefix = (compared_idx == 1) ? "top1_accuracy " : "mAP ";
  LOG(INFO) << "--- Accuracy summary --- ";
  LOG(INFO) << "Accepted " << prefix
            << "drop threshold: " << FLAGS_quantized_accuracy
            << ". (condition: (FP32_" << prefix << " - " << data_type_name
            << "_" << prefix << ") <= threshold)";
  LOG(INFO) << "FP32: avg " << prefix << std::fixed << std::setw(6)
            << std::setprecision(4) << avg_acc_ref;
  LOG(INFO) << data_type_name << ": avg " << prefix << std::fixed
            << std::setw(6) << std::setprecision(4) << avg_acc;
}

void SummarizePerformance(const char *title, float sample) {
  PADDLE_ENFORCE_GT(sample,
                    0.0,
                    common::errors::InvalidArgument(
                        "[Error info] sample must be greater than 0.0\n"
                        "[Argument info] The current sample is %f.",
                        sample));
  auto throughput = 1000.0 / sample;
  LOG(INFO) << title << ": avg fps: " << std::fixed << std::setw(6)
            << std::setprecision(4) << throughput << ", avg latency: " << sample
            << " ms";
}

void SummarizePerformance(const char *title_fp32,
                          float sample_latency_fp32,
                          const char *title,
                          float sample_latency) {
  if (FLAGS_enable_fp32) SummarizePerformance(title_fp32, sample_latency_fp32);
  if (FLAGS_enable_int8_ptq || FLAGS_enable_int8_qat || FLAGS_enable_bf16)
    SummarizePerformance(title, sample_latency);
}

float CompareAccuracyOne(
    const std::vector<std::vector<PaddleTensor>> &output_slots,
    int compared_idx) {
  PADDLE_ENFORCE_GT(output_slots.size(),
                    0,
                    common::errors::InvalidArgument(
                        "The accuracy vector is empty. The accuracy vector "
                        "size should be bigger than 0"));

  float total_accs{0};

  for (size_t i = 0; i < output_slots.size(); ++i) {
    switch (compared_idx) {
      case 1:
        PADDLE_ENFORCE_GE(
            output_slots[i].size(),
            2UL,
            common::errors::InvalidArgument(
                "To achieve top 1 accuracy, output_slots size "
                "must be bigger than or equal to 2, but now the size is %d",
                output_slots[i].size()));
        break;
      case 2:
        PADDLE_ENFORCE_GE(
            output_slots[i].size(),
            3UL,
            common::errors::InvalidArgument(
                "To achieve top 5 accuracy or mean Average "
                "Precision (mAP), output_slots size must be "
                "bigger than or equal to 3, but now the size is %d",
                output_slots[i].size()));
        break;
      default:
        throw std::invalid_argument(
            "CompareAccuracy: compared_idx is out of range.");
    }

    if (output_slots[i][compared_idx].lod.size() > 0)
      throw std::invalid_argument("CompareAccuracy: output has nonempty LoD.");

    if (output_slots[i][compared_idx].dtype != ::paddle::PaddleDType::FLOAT32)
      throw std::invalid_argument(
          "CompareAccuracy: output is of a wrong type.");

    total_accs +=
        *static_cast<float *>(output_slots[i][compared_idx].data.data());
  }

  return total_accs / output_slots.size();
}

void CompareAccuracy(
    const std::vector<std::vector<PaddleTensor>> &output_slots_quant,
    const std::vector<std::vector<PaddleTensor>> &output_slots_ref,
    int compared_idx) {
  if ((FLAGS_enable_fp32 &&
       (FLAGS_enable_int8_ptq || FLAGS_enable_int8_qat || FLAGS_enable_bf16)) &&
      (output_slots_quant.size() == 0 || output_slots_ref.size()) == 0)
    throw std::invalid_argument(
        "CompareAccuracy: output_slots vector is empty.");

  float avg_acc_quant = 0.0;
  float avg_acc_ref = 0.0;

  if (FLAGS_enable_int8_ptq || FLAGS_enable_int8_qat || FLAGS_enable_bf16)
    avg_acc_quant = CompareAccuracyOne(output_slots_quant, compared_idx);

  if (FLAGS_enable_fp32)
    avg_acc_ref = CompareAccuracyOne(output_slots_ref, compared_idx);

  SummarizeAccuracy(avg_acc_ref, avg_acc_quant, compared_idx);

  if (FLAGS_enable_fp32) {
    PADDLE_ENFORCE_GT(avg_acc_ref,
                      0.0,
                      common::errors::PreconditionNotMet(
                          "[Error info] avg_acc_ref must be greater than 0.0.\n"
                          "[Condition info] The current avg_acc_ref is %f.",
                          avg_acc_ref));
  }

  if (FLAGS_enable_int8_ptq || FLAGS_enable_int8_qat || FLAGS_enable_bf16) {
    PADDLE_ENFORCE_GT(
        avg_acc_quant,
        0.0,
        common::errors::PreconditionNotMet(
            "[Error info] avg_acc_quant must be greater than 0.0.\n"
            "[Condition info] The current avg_acc_quant is %f.",
            avg_acc_quant));
  }

  if (FLAGS_enable_fp32 &&
      (FLAGS_enable_int8_ptq || FLAGS_enable_int8_qat || FLAGS_enable_bf16)) {
    PADDLE_ENFORCE_LE(
        avg_acc_ref - avg_acc_quant,
        FLAGS_quantized_accuracy,
        common::errors::PreconditionNotMet(
            "[Error info] avg_acc_ref - avg_acc_quant must be less than or "
            "equal to FLAGS_quantized_accuracy.\n"
            "[Condition info] Please check your input data."));
  }
}

void CompareDeterministic(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  int batch_size = FLAGS_batch_size;
  int num_times = FLAGS_repeat;
  auto predictor = CreateTestPredictor(config, FLAGS_use_analysis);

  std::vector<PaddleTensor> warmup_outputs, outputs;
  // run num_times to Compare Deterministic Result.
  for (size_t j = 0; j < inputs.size(); j++) {
    // warmup run
    predictor->Run(inputs[j], &warmup_outputs, batch_size);
    for (int i = 0; i < num_times; i++) {
      predictor->Run(inputs[j], &outputs, batch_size);
      CompareResult(outputs, warmup_outputs);
    }
  }
}

void CompareNativeAndAnalysis(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  PrintConfig(config, true);
  std::vector<std::vector<PaddleTensor>> native_outputs, analysis_outputs;
  TestOneThreadPrediction(config, inputs, &native_outputs, false);
  TestOneThreadPrediction(config, inputs, &analysis_outputs, true);
  PADDLE_ENFORCE_GT(native_outputs.size(),
                    0,
                    common::errors::InvalidArgument(
                        "The native outputs is empty. The native outputs "
                        "vector size must be bigger than 0"));
  PADDLE_ENFORCE_GT(analysis_outputs.size(),
                    0,
                    common::errors::InvalidArgument(
                        "The analysis outputs is empty. The analysis outputs "
                        "vector size must be bigger than 0"));
  CompareResult(analysis_outputs.back(), native_outputs.back());
}

void CompareQuantizedAndAnalysis(
    const AnalysisConfig *config,
    const AnalysisConfig *qconfig,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    const int compared_idx = 1) {
  PADDLE_ENFORCE_GT(
      inputs.size(),
      0,
      common::errors::PreconditionNotMet("There is no input data provided."));
  PADDLE_ENFORCE_EQ(
      inputs[0][0].shape[0],
      FLAGS_batch_size,
      common::errors::InvalidArgument(
          "Input data has to be packed batch by batch. The batchsize is set to "
          "%d, but the real input is packed with batchsize = %d",
          FLAGS_batch_size,
          inputs[0][0].shape[0]));
  LOG(INFO) << "FP32 & INT8 prediction run: batch_size " << FLAGS_batch_size
            << ", warmup batch size " << FLAGS_warmup_batch_size << ".";

  LOG(INFO) << "--- FP32 prediction start ---";
  auto *cfg = reinterpret_cast<const PaddlePredictor::Config *>(config);
  PrintConfig(cfg, true);
  std::vector<std::vector<PaddleTensor>> analysis_outputs;
  float sample_latency_fp32{-1};

  if (FLAGS_enable_fp32) {
    TestOneThreadPrediction(cfg,
                            inputs,
                            &analysis_outputs,
                            true,
                            VarType::FP32,
                            &sample_latency_fp32);
  }

  LOG(INFO) << "--- INT8 prediction start ---";
  auto *qcfg = reinterpret_cast<const PaddlePredictor::Config *>(qconfig);
  PrintConfig(qcfg, true);
  std::vector<std::vector<PaddleTensor>> quantized_outputs;
  float sample_latency_int8{-1};

  if (FLAGS_enable_int8_ptq || FLAGS_enable_int8_qat) {
    TestOneThreadPrediction(qcfg,
                            inputs,
                            &quantized_outputs,
                            true,
                            VarType::INT8,
                            &sample_latency_int8);
  }
  SummarizePerformance(
      "FP32", sample_latency_fp32, "INT8", sample_latency_int8);

  if (FLAGS_with_accuracy_layer)
    CompareAccuracy(quantized_outputs, analysis_outputs, compared_idx);
}

void CompareBFloat16AndAnalysis(
    const AnalysisConfig *config,
    const AnalysisConfig *qconfig,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    const int compared_idx = 1) {
  PADDLE_ENFORCE_EQ(
      inputs[0][0].shape[0],
      FLAGS_batch_size,
      common::errors::InvalidArgument(
          "Input data has to be packed batch by batch. The batchsize is set to "
          "%d, but the real input is packed with batchsize = %d",
          FLAGS_batch_size,
          inputs[0][0].shape[0]));
  LOG(INFO) << "FP32 & BF16 prediction run: batch_size " << FLAGS_batch_size;

  LOG(INFO) << "--- FP32 prediction start ---";
  auto *cfg = reinterpret_cast<const PaddlePredictor::Config *>(config);
  PrintConfig(cfg, true);
  std::vector<std::vector<PaddleTensor>> analysis_outputs;
  float sample_latency_fp32{-1};

  if (FLAGS_enable_fp32) {
    TestOneThreadPrediction(cfg,
                            inputs,
                            &analysis_outputs,
                            true,
                            VarType::FP32,
                            &sample_latency_fp32);
  }

  LOG(INFO) << "--- BF16 prediction start ---";
  auto *qcfg = reinterpret_cast<const PaddlePredictor::Config *>(qconfig);
  PrintConfig(qcfg, true);
  std::vector<std::vector<PaddleTensor>> bf16_outputs;
  float sample_latency_bf16{-1};

  if (FLAGS_enable_bf16) {
    TestOneThreadPrediction(
        qcfg, inputs, &bf16_outputs, true, VarType::FP32, &sample_latency_bf16);
  }
  SummarizePerformance(
      "FP32", sample_latency_fp32, "BF16", sample_latency_bf16);

  if (FLAGS_with_accuracy_layer)
    CompareAccuracy(bf16_outputs, analysis_outputs, compared_idx);
}

void CompareAnalysisAndAnalysis(
    const AnalysisConfig *config1,
    const AnalysisConfig *config2,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    const bool with_accuracy_layer = FLAGS_with_accuracy_layer,
    const int compared_idx = 1) {
  PADDLE_ENFORCE_EQ(
      inputs[0][0].shape[0],
      FLAGS_batch_size,
      common::errors::InvalidArgument(
          "Input data has to be packed batch by batch. The batchsize is set to "
          "%d, but the real input is packed with batchsize = %d",
          FLAGS_batch_size,
          inputs[0][0].shape[0]));

  LOG(INFO) << "FP32 & INT8 prediction run: batch_size " << FLAGS_batch_size
            << ", warmup batch size " << FLAGS_warmup_batch_size << ".";

  LOG(INFO) << "--- FP32 prediction start ---";
  auto *cfg1 = reinterpret_cast<const PaddlePredictor::Config *>(config1);
  PrintConfig(cfg1, true);
  std::vector<std::vector<PaddleTensor>> analysis_outputs;
  float sample_latency_fp32{-1};

  if (FLAGS_enable_fp32) {
    TestOneThreadPrediction(cfg1,
                            inputs,
                            &analysis_outputs,
                            true,
                            VarType::FP32,
                            &sample_latency_fp32);
  }

  LOG(INFO) << "--- INT8 prediction start ---";
  auto *cfg2 = reinterpret_cast<const PaddlePredictor::Config *>(config2);
  PrintConfig(cfg2, true);
  std::vector<std::vector<PaddleTensor>> int8_outputs;
  float sample_latency_int8{-1};

  if (FLAGS_enable_int8_ptq || FLAGS_enable_int8_qat) {
    TestOneThreadPrediction(
        cfg2, inputs, &int8_outputs, true, VarType::INT8, &sample_latency_int8);
  }
  SummarizePerformance(
      "FP32", sample_latency_fp32, "INT8", sample_latency_int8);
  if (with_accuracy_layer) {
    CompareAccuracy(int8_outputs, analysis_outputs, compared_idx);
  }
}

void CompareNativeAndAnalysis(
    PaddlePredictor *native_pred,
    PaddlePredictor *analysis_pred,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  int batch_size = FLAGS_batch_size;
  std::vector<PaddleTensor> native_outputs, analysis_outputs;
  native_pred->Run(inputs[0], &native_outputs, batch_size);
  analysis_pred->Run(inputs[0], &analysis_outputs, batch_size);
  CompareResult(analysis_outputs, native_outputs);
}

void CompareAnalysisAndZeroCopy(
    PaddlePredictor::Config *config,
    PaddlePredictor::Config *config1,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    const std::vector<std::string> &outputs_name) {
  int batch_size = FLAGS_batch_size;
  // analysis
  std::vector<PaddleTensor> analysis_outputs;
  auto predictor = CreateTestPredictor(config, true);
  predictor->Run(inputs[0], &analysis_outputs, batch_size);
  // analysis + zero_copy
  std::vector<ZeroCopyTensor> zerocopy_outputs;
  predictor = CreateTestPredictor(config1, true);
  ConvertPaddleTensorToZeroCopyTensor(predictor.get(), inputs[0]);
  predictor->ZeroCopyRun();
  for (size_t i = 0; i < outputs_name.size(); i++) {
    ZeroCopyTensor zerocopy_output =
        *predictor->GetOutputTensor(outputs_name[i]).get();
    zerocopy_outputs.emplace_back(zerocopy_output);
    LOG(INFO) << "ZeroCopy output: " << DescribeZeroCopyTensor(zerocopy_output);
  }
  // compare
  CompareResult(analysis_outputs, zerocopy_outputs);
}

template <typename T>
std::string DenseTensorSummary(const phi::DenseTensor &tensor) {
  std::stringstream ss;
  ss << "\n---- tensor ---" << '\n';
  ss << "lod: [";
  for (const auto &level : tensor.lod()) {
    ss << "[ ";
    for (auto i : level) {
      ss << i << ", ";
    }
    ss << "]";
  }
  ss << "]\n";

  ss << "shape: [";
  int size = 1;
  for (int i = 0; i < tensor.dims().size(); i++) {
    int dim = tensor.dims()[i];
    ss << dim << ", ";
    size *= dim;
  }
  ss << "]\n";

  ss << "data: ";
  for (int i = 0; i < std::min(20, size); i++) {
    ss << tensor.data<T>()[i] << " ";
  }
  ss << "\n";

  return ss.str();
}

static bool CompareLoD(const phi::LegacyLoD &a, const phi::LegacyLoD &b) {
  if (a.size() != b.size()) {
    LOG(ERROR) << string::Sprintf(
        "lod size not match %d != %d", a.size(), b.size());
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    auto &al = a[i];
    auto &bl = b[i];
    if (al.size() != bl.size()) {
      LOG(ERROR) << string::Sprintf(
          "level size %d != %d", al.size(), bl.size());
      return false;
    }
  }
  return true;
}

static bool CompareShape(const std::vector<int64_t> &a,
                         const std::vector<int64_t> &b) {
  if (a.size() != b.size()) {
    LOG(ERROR) << string::Sprintf(
        "shape size not match %d != %d", a.size(), b.size());
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      LOG(ERROR) << string::Sprintf(
          "shape %d-th element not match %d != %d", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

static bool CompareTensorData(const phi::DenseTensor &a,
                              const phi::DenseTensor &b) {
  auto a_shape = common::vectorize(a.dims());
  auto b_shape = common::vectorize(b.dims());
  size_t a_size = std::accumulate(
      a_shape.begin(), a_shape.end(), size_t{1}, [](int a, int b) {
        return a * b;
      });
  size_t b_size = std::accumulate(
      b_shape.begin(), b_shape.end(), size_t{1}, [](int a, int b) {
        return a * b;
      });
  if (a_size != b_size) {
    LOG(ERROR) << string::Sprintf(
        "tensor data size not match, %d != %d", a_size, b_size);
  }

  for (size_t i = 0; i < a_size; i++) {
    if (framework::TransToProtoVarType(a.dtype()) == VarType::FP32) {
      const auto *a_data = a.data<float>();
      const auto *b_data = b.data<float>();
      if (std::abs(a_data[i] - b_data[i]) > 1e-3) {
        LOG(ERROR) << string::Sprintf(
            "tensor data %d-th element not match, %f != %f",
            i,
            a_data[i],
            b_data[i]);
        return false;
      }
    } else if (framework::TransToProtoVarType(a.dtype()) == VarType::INT64) {
      const auto *a_data = a.data<int64_t>();
      const auto *b_data = b.data<int64_t>();
      if (std::abs(a_data[i] - b_data[i]) > 1e-3) {
        LOG(ERROR) << string::Sprintf(
            "tensor data %d-th element not match, %f != %f",
            i,
            a_data[i],
            b_data[i]);
        return false;
      }
    }
  }

  return true;
}

static bool CompareTensor(const phi::DenseTensor &a,
                          const phi::DenseTensor &b) {
  if (!CompareLoD(a.lod(), b.lod())) {
    return false;
  }
  if (!CompareShape(common::vectorize(a.dims()), common::vectorize(b.dims()))) {
    return false;
  }

  if (!CompareTensorData(a, b)) {
    return false;
  }

  return true;
}

void ConvertFP32toFP16(::paddle::PaddleTensor &tensor  // NOLINT
) {
  int num = 1;
  for (auto dim : tensor.shape) {
    num *= dim;
  }
  PADDLE_ENFORCE_EQ(
      tensor.dtype,
      PaddleDType::FLOAT32,
      common::errors::InvalidArgument(
          "The tensor dtype is not float32, only support float32 as input"));
  float *fp32_data = reinterpret_cast<float *>(tensor.data.data());
  float16 *fp16_data = new float16[num];
  for (int i = 0; i < num; i++) {
    fp16_data[i] = float16(fp32_data[i]);
  }
  tensor.data =
      PaddleBuf(static_cast<void *>(fp16_data), num * sizeof(float16));
  tensor.dtype = PaddleDType::FLOAT16;
}

void ConvertFP16toFP32(::paddle::PaddleTensor &tensor  // NOLINT
) {
  int num = 1;
  for (auto dim : tensor.shape) {
    num *= dim;
  }
  PADDLE_ENFORCE_EQ(
      tensor.dtype,
      PaddleDType::FLOAT16,
      common::errors::InvalidArgument(
          "The tensor dtype is not float16, only support float16 as input"));
  float16 *fp16_data = reinterpret_cast<float16 *>(tensor.data.data());
  float *fp32_data = new float[num];
  for (int i = 0; i < num; i++) {
    fp32_data[i] = static_cast<float>(fp16_data[i]);
  }
  tensor.data = PaddleBuf(static_cast<void *>(fp32_data), num * sizeof(float));
  tensor.dtype = PaddleDType::FLOAT32;
}

}  // namespace inference
}  // namespace paddle
