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

#include <thread>  // NOLINT

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "test/cpp/inference/test_helper.h"

#ifdef __clang__
#define ACC_DIFF 4e-3
#else
#define ACC_DIFF 2e-3
#endif

PD_DEFINE_string(word2vec_dirname,
                 "",
                 "Directory of the word2vec inference model.");
PD_DEFINE_string(book_dirname, "", "Directory of the book inference model.");

namespace paddle {

PaddleTensor LodTensorToPaddleTensor(phi::DenseTensor* t) {
  PaddleTensor pt;

  if (framework::TransToProtoVarType(t->dtype()) ==
      framework::proto::VarType::INT64) {
    pt.data.Reset(t->data(), t->numel() * sizeof(int64_t));
    pt.dtype = PaddleDType::INT64;
  } else if (framework::TransToProtoVarType(t->dtype()) ==
             framework::proto::VarType::FP32) {
    pt.data.Reset(t->data(), t->numel() * sizeof(float));
    pt.dtype = PaddleDType::FLOAT32;
  } else if (framework::TransToProtoVarType(t->dtype()) ==
             framework::proto::VarType::INT32) {
    pt.data.Reset(t->data(), t->numel() * sizeof(int32_t));
    pt.dtype = PaddleDType::INT32;
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported tensor date type. Now only supports INT64, FP32, INT32."));
  }
  pt.shape = common::vectorize<int>(t->dims());
  return pt;
}

NativeConfig GetConfig() {
  NativeConfig config;
  config.model_dir = FLAGS_word2vec_dirname;
  LOG(INFO) << "dirname  " << config.model_dir;
  config.fraction_of_gpu_memory = 0.15;
  config.device = 0;
  return config;
}

void MainWord2Vec(const ::paddle::PaddlePlace& place) {
  NativeConfig config = GetConfig();
  auto predictor = CreatePaddlePredictor<NativeConfig>(config);
  config.use_gpu = ::paddle::gpu_place_used(place);
  config.use_xpu = ::paddle::xpu_place_used(place);

  phi::DenseTensor first_word, second_word, third_word, fourth_word;
  phi::LegacyLoD lod{{0, 1}};
  int64_t dict_size = 2073;  // The size of dictionary

  SetupDenseTensor(&first_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupDenseTensor(&second_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupDenseTensor(&third_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupDenseTensor(&fourth_word, lod, static_cast<int64_t>(0), dict_size - 1);

  std::vector<PaddleTensor> paddle_tensor_feeds;
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&first_word));
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&second_word));
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&third_word));
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&fourth_word));

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));
  ASSERT_EQ(outputs.size(), 1UL);
  size_t len = outputs[0].data.length();
  float* data = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < len / sizeof(float); ++j) {
    ASSERT_LT(data[j], 1.0);
    ASSERT_GT(data[j], -1.0);
  }

  std::vector<phi::DenseTensor*> cpu_feeds;
  cpu_feeds.push_back(&first_word);
  cpu_feeds.push_back(&second_word);
  cpu_feeds.push_back(&third_word);
  cpu_feeds.push_back(&fourth_word);

  framework::FetchType output1;
  std::vector<::paddle::framework::FetchType*> cpu_fetches1;
  cpu_fetches1.push_back(&output1);

  TestInference<phi::CPUPlace>(config.model_dir, cpu_feeds, cpu_fetches1);

  auto output1_tensor = PADDLE_GET(phi::DenseTensor, output1);
  float* lod_data = output1_tensor.data<float>();
  for (int i = 0; i < output1_tensor.numel(); ++i) {
    EXPECT_LT(lod_data[i] - data[i], ACC_DIFF);
    EXPECT_GT(lod_data[i] - data[i], -ACC_DIFF);
  }
}

void MainImageClassification(const ::paddle::PaddlePlace& place) {
  int batch_size = 2;
  bool repeat = false;
  NativeConfig config = GetConfig();
  config.use_gpu = ::paddle::gpu_place_used(place);
  config.use_xpu = ::paddle::xpu_place_used(place);
  config.model_dir =
      FLAGS_book_dirname + "/image_classification_resnet.inference.model";

  const bool is_combined = false;
  std::vector<std::vector<int64_t>> feed_target_shapes =
      GetFeedTargetShapes(config.model_dir, is_combined);

  phi::DenseTensor input;
  // Use normilized image pixels as input data,
  // which should be in the range [0.0, 1.0].
  feed_target_shapes[0][0] = batch_size;
  phi::DDim input_dims = common::make_ddim(feed_target_shapes[0]);
  SetupTensor<float>(
      &input, input_dims, static_cast<float>(0), static_cast<float>(1));
  std::vector<phi::DenseTensor*> cpu_feeds;
  cpu_feeds.push_back(&input);

  framework::FetchType output1;
  std::vector<framework::FetchType*> cpu_fetches1;
  cpu_fetches1.push_back(&output1);

  TestInference<phi::CPUPlace, false, true>(
      config.model_dir, cpu_feeds, cpu_fetches1, repeat, is_combined);

  auto predictor = CreatePaddlePredictor(config);
  std::vector<PaddleTensor> paddle_tensor_feeds;
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&input));

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));
  ASSERT_EQ(outputs.size(), 1UL);
  size_t len = outputs[0].data.length();
  float* data = static_cast<float*>(outputs[0].data.data());
  float* lod_data = PADDLE_GET(phi::DenseTensor, output1).data<float>();
  for (size_t j = 0; j < len / sizeof(float); ++j) {
    EXPECT_NEAR(lod_data[j], data[j], ACC_DIFF);
  }
}

void MainThreadsWord2Vec(const ::paddle::PaddlePlace& place) {
  NativeConfig config = GetConfig();
  config.use_gpu = ::paddle::gpu_place_used(place);
  config.use_xpu = ::paddle::xpu_place_used(place);
  auto main_predictor = CreatePaddlePredictor<NativeConfig>(config);

  // prepare inputs data and reference results
  constexpr int num_jobs = 3;
  std::vector<std::vector<phi::DenseTensor>> jobs(num_jobs);
  std::vector<std::vector<PaddleTensor>> paddle_tensor_feeds(num_jobs);
  std::vector<framework::FetchType> refs(num_jobs);
  for (size_t i = 0; i < jobs.size(); ++i) {
    // each job has 4 words
    jobs[i].resize(4);
    for (size_t j = 0; j < 4; ++j) {
      phi::LegacyLoD lod{{0, 1}};
      int64_t dict_size = 2073;  // The size of dictionary
      SetupDenseTensor(
          &jobs[i][j], lod, static_cast<int64_t>(0), dict_size - 1);
      paddle_tensor_feeds[i].push_back(LodTensorToPaddleTensor(&jobs[i][j]));
    }

    // get reference result of each job
    std::vector<phi::DenseTensor*> ref_feeds;
    std::vector<::paddle::framework::FetchType*> ref_fetches(1, &refs[i]);
    for (auto& word : jobs[i]) {
      ref_feeds.push_back(&word);
    }
    TestInference<phi::CPUPlace>(config.model_dir, ref_feeds, ref_fetches);
  }

  // create threads and each thread run 1 job
  std::vector<std::thread> threads;
  for (int tid = 0; tid < num_jobs; ++tid) {
    threads.emplace_back([&, tid]() {
      auto predictor = CreatePaddlePredictor(config);
      auto& local_inputs = paddle_tensor_feeds[tid];
      std::vector<PaddleTensor> local_outputs;
      ASSERT_TRUE(predictor->Run(local_inputs, &local_outputs));

      // check outputs range
      ASSERT_EQ(local_outputs.size(), 1UL);
      const size_t len = local_outputs[0].data.length();
      float* data = static_cast<float*>(local_outputs[0].data.data());
      for (size_t j = 0; j < len / sizeof(float); ++j) {
        ASSERT_LT(data[j], 1.0);
        ASSERT_GT(data[j], -1.0);
      }

      // check outputs correctness
      auto ref_tensor = PADDLE_GET(phi::DenseTensor, refs[tid]);
      float* ref_data = ref_tensor.data<float>();
      EXPECT_EQ(ref_tensor.numel(), static_cast<int64_t>(len / sizeof(float)));
      for (int i = 0; i < ref_tensor.numel(); ++i) {
        EXPECT_NEAR(ref_data[i], data[i], 2e-3);
      }
    });
  }
  for (int i = 0; i < num_jobs; ++i) {
    threads[i].join();
  }
}

void MainThreadsImageClassification(const ::paddle::PaddlePlace& place) {
  constexpr int num_jobs = 4;  // each job run 1 batch
  constexpr int batch_size = 1;
  NativeConfig config = GetConfig();
  config.use_gpu = ::paddle::gpu_place_used(place);
  config.use_xpu = ::paddle::xpu_place_used(place);
  config.model_dir =
      FLAGS_book_dirname + "/image_classification_resnet.inference.model";

  auto main_predictor = CreatePaddlePredictor<NativeConfig>(config);
  std::vector<phi::DenseTensor> jobs(num_jobs);
  std::vector<std::vector<PaddleTensor>> paddle_tensor_feeds(num_jobs);
  std::vector<framework::FetchType> refs(num_jobs);
  for (size_t i = 0; i < jobs.size(); ++i) {
    // prepare inputs
    std::vector<std::vector<int64_t>> feed_target_shapes =
        GetFeedTargetShapes(config.model_dir, /*is_combined*/ false);
    feed_target_shapes[0][0] = batch_size;
    phi::DDim input_dims = common::make_ddim(feed_target_shapes[0]);
    SetupTensor<float>(&jobs[i], input_dims, 0.f, 1.f);
    paddle_tensor_feeds[i].push_back(LodTensorToPaddleTensor(&jobs[i]));

    // get reference result of each job
    std::vector<phi::DenseTensor*> ref_feeds(1, &jobs[i]);
    std::vector<framework::FetchType*> ref_fetches(1, &refs[i]);
    TestInference<phi::CPUPlace>(config.model_dir, ref_feeds, ref_fetches);
  }

  // create threads and each thread run 1 job
  std::vector<std::thread> threads;
  for (int tid = 0; tid < num_jobs; ++tid) {
    threads.emplace_back([&, tid]() {
      auto predictor = CreatePaddlePredictor(config);
      auto& local_inputs = paddle_tensor_feeds[tid];
      std::vector<PaddleTensor> local_outputs;
      ASSERT_TRUE(predictor->Run(local_inputs, &local_outputs));

      // check outputs correctness
      ASSERT_EQ(local_outputs.size(), 1UL);
      const size_t len = local_outputs[0].data.length();
      float* data = static_cast<float*>(local_outputs[0].data.data());
      auto ref_tensor = PADDLE_GET(phi::DenseTensor, refs[tid]);
      float* ref_data = ref_tensor.data<float>();
      EXPECT_EQ((size_t)ref_tensor.numel(), len / sizeof(float));
      for (int i = 0; i < ref_tensor.numel(); ++i) {
        EXPECT_NEAR(ref_data[i], data[i], ACC_DIFF);
      }
    });
  }
  for (int i = 0; i < num_jobs; ++i) {
    threads[i].join();
  }
}

TEST(inference_api_native, word2vec_cpu) {
  MainWord2Vec(::paddle::PaddlePlace::kCPU);
}
TEST(inference_api_native, word2vec_cpu_threads) {
  MainThreadsWord2Vec(::paddle::PaddlePlace::kCPU);
}
TEST(inference_api_native, image_classification_cpu) {
  MainImageClassification(::paddle::PaddlePlace::kCPU);
}
TEST(inference_api_native, image_classification_cpu_threads) {
  MainThreadsImageClassification(::paddle::PaddlePlace::kCPU);
}

#ifdef PADDLE_WITH_XPU
TEST(inference_api_native, word2vec_xpu) {
  MainWord2Vec(::paddle::PaddlePlace::kXPU);
}
TEST(inference_api_native, image_classification_xpu) {
  MainImageClassification(::paddle::PaddlePlace::kXPU);
}
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(inference_api_native, word2vec_gpu) {
  MainWord2Vec(::paddle::PaddlePlace::kGPU);
}
// Turn off temporarily for the unstable result.
// TEST(inference_api_native, word2vec_gpu_threads) {
//   MainThreadsWord2Vec(::paddle::PaddlePlace::kGPU);
// }
TEST(inference_api_native, image_classification_gpu) {
  MainImageClassification(::paddle::PaddlePlace::kGPU);
}
// Turn off temporarily for the unstable result.
// TEST(inference_api_native, image_classification_gpu_threads) {
//   MainThreadsImageClassification(::paddle::PaddlePlace::kGPU);
// }
#endif

#ifdef PADDLE_WITH_DNNL
TEST(inference_api_native, image_classification_cpu_onednn) {
  FLAGS_use_mkldnn = true;
  MainImageClassification(::paddle::PaddlePlace::kCPU);
}

TEST(inference_api_native, word2vec_cpu_onednn) {
  FLAGS_use_mkldnn = true;
  MainWord2Vec(::paddle::PaddlePlace::kCPU);
}
#endif

TEST(PassBuilder, Delete) {
  AnalysisConfig config;
  config.DisableGpu();
  config.pass_builder()->DeletePass("attention_lstm_fuse_pass");
  const auto& passes = config.pass_builder()->AllPasses();
  auto it = std::find(passes.begin(), passes.end(), "attention_lstm_fuse_pass");
  ASSERT_EQ(it, passes.end());
}

}  // namespace paddle
