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
#pragma once

#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/phi/common/port.h"
#include "paddle/phi/core/platform/profiler.h"

COMMON_DECLARE_bool(use_mkldnn);

namespace paddle {
bool gpu_place_used(const paddle::PaddlePlace& place) {
  return place == paddle::PaddlePlace::kGPU;
}
bool xpu_place_used(const paddle::PaddlePlace& place) {
  return place == paddle::PaddlePlace::kXPU;
}
bool cpu_place_used(const paddle::PaddlePlace& place) {
  return place == paddle::PaddlePlace::kCPU;
}
}  // namespace paddle

template <typename T>
void SetupTensor(phi::DenseTensor* input, phi::DDim dims, T lower, T upper) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  T* input_ptr = input->mutable_data<T>(dims, phi::CPUPlace());
  for (int i = 0; i < input->numel(); ++i) {
    input_ptr[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
  }
}

template <typename T>
void SetupTensor(phi::DenseTensor* input,
                 phi::DDim dims,
                 const std::vector<T>& data) {
  PADDLE_ENFORCE_EQ(common::product(dims),
                    static_cast<int64_t>(data.size()),
                    common::errors::InvalidArgument(
                        "common::product(dims) and data.size() are not equal"
                        "common::product(dims) is %d and data.size() is %d",
                        common::product(dims),
                        static_cast<int64_t>(data.size())));
  T* input_ptr = input->mutable_data<T>(dims, phi::CPUPlace());
  memcpy(input_ptr, data.data(), input->numel() * sizeof(T));
}

template <typename T>
void SetupDenseTensor(phi::DenseTensor* input,
                      const phi::LegacyLoD& lod,
                      T lower,
                      T upper) {
  input->set_lod(lod);
  int dim = lod[0][lod[0].size() - 1];
  SetupTensor<T>(input, {dim, 1}, lower, upper);
}

template <typename T>
void SetupDenseTensor(phi::DenseTensor* input,
                      phi::DDim dims,
                      const phi::LegacyLoD lod,
                      const std::vector<T>& data) {
  const size_t level = lod.size() - 1;
  PADDLE_ENFORCE_EQ(dims[0],
                    static_cast<int64_t>((lod[level]).back()),
                    common::errors::InvalidArgument(
                        "dims[0] is not equal with (lod[level]).back()"
                        "while dims[0] is %d and (lod[level]).back() is %d",
                        dims[0],
                        static_cast<int64_t>((lod[level]).back())));
  input->set_lod(lod);
  SetupTensor<T>(input, dims, data);
}

template <typename T>
void CheckError(const phi::DenseTensor& output1,
                const phi::DenseTensor& output2) {
  // Check lod information
  EXPECT_EQ(output1.lod(), output2.lod());

  EXPECT_EQ(output1.dims(), output2.dims());
  EXPECT_EQ(output1.numel(), output2.numel());

  T err = static_cast<T>(0);
  if (typeid(T) == typeid(float)) {
    err = 1E-3;
  } else if (typeid(T) == typeid(double)) {
    err = 1E-6;
  } else {
    err = 0;
  }

  size_t count = 0;
  for (int64_t i = 0; i < output1.numel(); ++i) {
    if (fabs(output1.data<T>()[i] - output2.data<T>()[i]) > err) {
      count++;
    }
  }
  EXPECT_EQ(count, 0U) << "There are " << count << " different elements.";
}

std::unique_ptr<paddle::framework::ProgramDesc> InitProgram(
    paddle::framework::Executor* executor,
    paddle::framework::Scope* scope,
    const std::string& dirname,
    const bool is_combined = false,
    const std::string& prog_filename = "__model_combined__",
    const std::string& param_filename = "__params_combined__") {
  std::unique_ptr<paddle::framework::ProgramDesc> inference_program;
  if (is_combined) {
    // All parameters are saved in a single file.
    // Hard-coding the file names of program and parameters in unittest.
    // The file names should be consistent with that used in Python API
    //  `fluid.io.save_inference_model`.
    inference_program = paddle::inference::Load(executor,
                                                scope,
                                                dirname + "/" + prog_filename,
                                                dirname + "/" + param_filename);
  } else {
    // Parameters are saved in separate files sited in the specified
    // `dirname`.
    inference_program = paddle::inference::Load(executor, scope, dirname);
  }
  return inference_program;
}

std::vector<std::vector<int64_t>> GetFeedTargetShapes(
    const std::string& dirname,
    const bool is_combined = false,
    const std::string& prog_filename = "__model_combined__",
    const std::string& param_filename = "__params_combined__") {
  auto place = phi::CPUPlace();
  auto executor = paddle::framework::Executor(place);
  auto* scope = new paddle::framework::Scope();

  auto inference_program = InitProgram(
      &executor, scope, dirname, is_combined, prog_filename, param_filename);
  auto& global_block = inference_program->Block(0);

  const std::vector<std::string>& feed_target_names =
      inference_program->GetFeedTargetNames();
  std::vector<std::vector<int64_t>> feed_target_shapes;
  for (size_t i = 0; i < feed_target_names.size(); ++i) {
    auto* var = global_block.FindVar(feed_target_names[i]);
    std::vector<int64_t> var_shape = var->GetShape();
    feed_target_shapes.push_back(var_shape);
  }

  delete scope;
  return feed_target_shapes;
}

template <typename Place, bool CreateVars = true, bool PrepareContext = false>
void TestInference(
    const std::string& dirname,
    const std::vector<phi::DenseTensor*>& cpu_feeds,
    const std::vector<paddle::framework::FetchType*>& cpu_fetches,
    const int repeat = 1,
    const bool is_combined = false) {
  // 1. Define place, executor, scope
  auto place = Place();
  auto executor = paddle::framework::Executor(place);
  auto* scope = new paddle::framework::Scope();

  // Profile the performance
  paddle::platform::ProfilerState state;
  if (phi::is_cpu_place(place)) {
    state = paddle::platform::ProfilerState::kCPU;
  } else {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    state = paddle::platform::ProfilerState::kAll;
    // The default device_id of phi::GPUPlace is 0.
    // Users can get the device_id using:
    //   int device_id = place.GetDeviceId();
    paddle::platform::SetDeviceId(0);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "'CUDAPlace' is not supported in CPU only device."));
#endif
  }

  // 2. Initialize the inference_program and load parameters
  std::unique_ptr<paddle::framework::ProgramDesc> inference_program;

  // Enable the profiler
  paddle::platform::EnableProfiler(state);
  {
    phi::RecordEvent record_event("init_program");
    inference_program = InitProgram(&executor, scope, dirname, is_combined);
  }

  // Disable the profiler and print the timing information
  paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kDefault,
                                    "load_program_profiler");
  paddle::platform::ResetProfiler();

  // 3. Get the feed_target_names and fetch_target_names
  const std::vector<std::string>& feed_target_names =
      inference_program->GetFeedTargetNames();
  const std::vector<std::string>& fetch_target_names =
      inference_program->GetFetchTargetNames();

  // 4. Prepare inputs: set up maps for feed targets
  std::map<std::string, const phi::DenseTensor*> feed_targets;
  for (size_t i = 0; i < feed_target_names.size(); ++i) {
    // Please make sure that cpu_feeds[i] is right for feed_target_names[i]
    feed_targets[feed_target_names[i]] = cpu_feeds[i];
  }

  // 5. Define Tensor to get the outputs: set up maps for fetch targets
  std::map<std::string, paddle::framework::FetchType*> fetch_targets;
  for (size_t i = 0; i < fetch_target_names.size(); ++i) {
    fetch_targets[fetch_target_names[i]] = cpu_fetches[i];
  }

  // 6. If export Flags_use_mkldnn=True, use onednn related ops.
  if (FLAGS_use_mkldnn) executor.EnableMKLDNN(*inference_program);

  // 7. Run the inference program
  {
    if (!CreateVars) {
      // If users don't want to create and destroy variables every time they
      // run, they need to set `create_vars` to false and manually call
      // `CreateVariables` before running.
      executor.CreateVariables(*inference_program, scope, 0);
    }

    // Ignore the profiling results of the first run
    std::unique_ptr<paddle::framework::ExecutorPrepareContext> ctx;
    bool CreateLocalScope = CreateVars;
    if (PrepareContext) {
      ctx = executor.Prepare(*inference_program, 0);
      executor.RunPreparedContext(ctx.get(),
                                  scope,
                                  &feed_targets,
                                  &fetch_targets,
                                  CreateLocalScope,
                                  CreateVars);
    } else {
      executor.Run(*inference_program,
                   scope,
                   &feed_targets,
                   &fetch_targets,
                   CreateLocalScope,
                   CreateVars);
    }

    // Enable the profiler
    paddle::platform::EnableProfiler(state);

    // Run repeat times to profile the performance
    for (int i = 0; i < repeat; ++i) {
      phi::RecordEvent record_event("run_inference");

      if (PrepareContext) {
        // Note: if you change the inference_program, you need to call
        // executor.Prepare() again to get a new ExecutorPrepareContext.
        executor.RunPreparedContext(ctx.get(),
                                    scope,
                                    &feed_targets,
                                    &fetch_targets,
                                    CreateLocalScope,
                                    CreateVars);
      } else {
        executor.Run(*inference_program,
                     scope,
                     &feed_targets,
                     &fetch_targets,
                     CreateLocalScope,
                     CreateVars);
      }
    }

    // Disable the profiler and print the timing information
    paddle::platform::DisableProfiler(
        paddle::platform::EventSortingKey::kDefault, "run_inference_profiler");
    paddle::platform::ResetProfiler();
  }

  delete scope;
}
