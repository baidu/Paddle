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

#include <gtest/gtest.h>
#include "gflags/gflags.h"
#include "paddle/fluid/inference/tests/test_helper.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(dirname, "", "Directory of the inference model.");
DEFINE_bool(combine,
            false,
            "To run the test with load_combine and save_combine");
DEFINE_int32(batchsize, 1, "Batch size of input");

void with_combine() {
  using paddle::platform::DeviceContext;
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::CUDAPlace;
  using paddle::platform::Event;
  using paddle::platform::EventKind;

  // LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // In unittests, this is done in paddle/testing/paddle_gtest_main.cc

  // Add a start_event: setting up input
  Event start_event1(EventKind::kPushRange, "setup_input", 0, nullptr);

  paddle::framework::LoDTensor input;
  // Use normilized image pixels as input data,
  // which should be in the range [-1.0, 1.0].
  SetupTensor<float>(input,
                     {FLAGS_batchsize, 1, 28, 28},
                     static_cast<float>(-1),
                     static_cast<float>(1));
  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&input);

  paddle::framework::LoDTensor output1;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  Event stop_event1(EventKind::kPopRange, "setup_input", 0, nullptr);
  LOG(INFO) << "Setting_input: " << start_event1.CpuElapsedMs(stop_event1)
            << std::endl;

  // Run inference on CPU
  TestInference<paddle::platform::CPUPlace, true>(
      dirname, cpu_feeds, cpu_fetchs1);
// LOG(INFO) << output1.dims();

#ifdef PADDLE_WITH_CUDA
  paddle::framework::LoDTensor output2;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
  cpu_fetchs2.push_back(&output2);

  // Run inference on CUDA GPU
  TestInference<paddle::platform::CUDAPlace, true>(
      dirname, cpu_feeds, cpu_fetchs2);
  // LOG(INFO) << output2.dims();

  CheckError<float>(output1, output2);
#endif
}

void separate() {
  using paddle::platform::DeviceContext;
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::CUDAPlace;
  using paddle::platform::Event;
  using paddle::platform::EventKind;

  // LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // In unittests, this is done in paddle/testing/paddle_gtest_main.cc

  int64_t batch_size = FLAGS_batchsize;

  // Add a start_event: setting up input
  Event start_event1(EventKind::kPushRange, "setup_input", 0, nullptr);

  paddle::framework::LoDTensor input;
  // Use normilized image pixels as input data,
  // which should be in the range [-1.0, 1.0].
  SetupTensor<float>(input,
                     {batch_size, 1, 28, 28},
                     static_cast<float>(-1),
                     static_cast<float>(1));
  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&input);

  paddle::framework::LoDTensor output1;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  Event stop_event1(EventKind::kPopRange, "setup_input", 0, nullptr);
  LOG(INFO) << "Setting_input: " << start_event1.CpuElapsedMs(stop_event1)
            << std::endl;

  // Run inference on CPU
  // Event start_event2(EventKind::kPushRange, "run_inference", 0, nullptr);
  TestInference<paddle::platform::CPUPlace>(dirname, cpu_feeds, cpu_fetchs1);
// Event stop_event2(EventKind::kPopRange, "run_inference", 0, nullptr);
// LOG(INFO) << "Running_inference: " <<
// start_event2.CpuElapsedMs(stop_event2)
//           << std::endl;
// LOG(INFO) << "Running inference: " <<
// start_event2.CpuElapsedMs(stop_event2)
//          << std::endl;

// LOG(INFO) << output1.dims();

#ifdef PADDLE_WITH_CUDA
  paddle::framework::LoDTensor output2;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
  cpu_fetchs2.push_back(&output2);

  //  DeviceContext* dev_ctx = new CUDADeviceContext(CUDAPlace(0));
  //  Event start_event_gpu1(EventKind::kPushRange, "gpu_run_inf", 0, dev_ctx);
  //  EXPECT_TRUE(start_event_gpu1.has_cuda() == true);

  // Run inference on CUDA GPU
  TestInference<paddle::platform::CUDAPlace>(dirname, cpu_feeds, cpu_fetchs2);
  //  Event stop_event_gpu1(EventKind::kPopRange, "gpu_run_inf", 0, dev_ctx);
  //  LOG(INFO) << "Running_inference_gpu: "
  //            << start_event_gpu1.CudaElapsedMs(stop_event_gpu1) << std::endl;
  // LOG(INFO) << output2.dims();

  CheckError<float>(output1, output2);
#endif
}

TEST(inference, recognize_digits_combine) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  if (FLAGS_combine) {
    LOG(INFO) << "RUNNING WITH COMBINE" << std::endl;
    for (int i = 0; i < 10; i++) {
      with_combine();
    }
  }
}

TEST(inference, recognize_digits) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  if (!FLAGS_combine) {
    LOG(INFO) << "RUNNING WITH SEPARATE" << std::endl;
    for (int i = 0; i < 10; i++) {
      separate();
    }
  }
}
