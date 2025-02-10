// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/runtime/cuda/cuda_module.h"

#include <gtest/gtest.h>

#include <random>

#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"
#include "paddle/cinn/runtime/cuda/test_util.h"
#include "paddle/cinn/runtime/cuda/use_extern_funcs.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace runtime {
namespace cuda {

TEST(CUDAModule, basic) {
  backends::nvrtc::Compiler compiler;

  std::string source_code = R"ROC(
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = a * x[tid] + y[tid];
  }
}
)ROC";

  auto ptx = compiler(source_code);
  PADDLE_ENFORCE_NE(
      ptx.empty(), true, ::common::errors::NotFound("ptx is empty!"));

  CUDAModule module(ptx, CUDAModule::Kind::PTX);
  auto func = module.GetFunction(0, "saxpy");
  ASSERT_TRUE(func);
}

TEST(CUDAModule, float16) {
  using cinn::common::float16;
  using runtime::cuda::util::Vector;

  auto generate_ptx = [] {
    backends::nvrtc::Compiler compiler;

    std::string source_code = R"(
  #include <cstdint>
  #define CINN_WITH_CUDA
  #include "float16.h"
  using cinn::common::float16;

  extern "C" __global__
  void cast_fp32_to_fp16_cuda_kernel(const float* input, const int num, float16* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < num) {
      output[idx] = float16(input[idx]);
    }
  }
  )";

    auto ptx = compiler(source_code);
    PADDLE_ENFORCE_NE(
        ptx.empty(), true, ::common::errors::NotFound("ptx is empty!"));
    return ptx;
  };

  auto ptx = generate_ptx();

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);
  auto func = cuda_module.GetFunction(0, "cast_fp32_to_fp16_cuda_kernel");
  ASSERT_TRUE(func);

  int size = 100;
  dim3 blocks_per_grid(1);
  dim3 threads_per_block(100);

  std::vector<float> x_host(size);
  {
    std::random_device r;
    std::default_random_engine eng(r());
    std::uniform_real_distribution<float> dis(1e-5f, 1.0f);
    for (size_t i = 0; i < x_host.size(); ++i) {
      x_host[i] = dis(eng);
    }
  }
  Vector<float> x_device(x_host);
  Vector<float16> y_device(size);
  auto* x_p{x_device.data()};
  auto* y_p{y_device.data()};

  void* args[] = {&x_p, &size, &y_p};
  cuda_module.LaunchKernel(0,
                           "cast_fp32_to_fp16_cuda_kernel",
                           blocks_per_grid,
                           threads_per_block,
                           args);
  CUDA_CALL(cudaDeviceSynchronize());

  std::vector<float16> y_host = y_device.to_host();
  bool res = std::equal(x_host.begin(),
                        x_host.end(),
                        y_host.begin(),
                        [](float x, float16 y) -> bool {
                          return std::abs(x - static_cast<float>(y)) < 1e-2f;
                        });
  PADDLE_ENFORCE_EQ(
      res,
      true,
      ::common::errors::PreconditionNotMet(
          "The difference between two arrays exceeds the bound."));
}

TEST(CUDAModule, bfloat16) {
  using cinn::common::bfloat16;
  using runtime::cuda::util::Vector;

  auto generate_ptx = [] {
    backends::nvrtc::Compiler compiler;

    std::string source_code = R"(
  #include <cstdint>
  #define CINN_WITH_CUDA
  #include "bfloat16.h"
  using cinn::common::bfloat16;

  extern "C" __global__
  void cast_fp32_to_bf16_cuda_kernel(const float* input, const int num, bfloat16* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < num) {
      output[idx] = bfloat16(input[idx]);
    }
  }
  )";

    auto ptx = compiler(source_code);
    PADDLE_ENFORCE_NE(
        ptx.empty(), true, ::common::errors::NotFound("ptx is empty!"));
    return ptx;
  };

  auto ptx = generate_ptx();

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);
  auto func = cuda_module.GetFunction(0, "cast_fp32_to_bf16_cuda_kernel");
  ASSERT_TRUE(func);

  int size = 100;
  dim3 blocks_per_grid(1);
  dim3 threads_per_block(100);

  std::vector<float> x_host(size);
  {
    std::random_device r;
    std::default_random_engine eng(r());
    std::uniform_real_distribution<float> dis(1e-5f, 1.0f);
    for (size_t i = 0; i < x_host.size(); ++i) {
      x_host[i] = dis(eng);
    }
  }
  Vector<float> x_device(x_host);
  Vector<bfloat16> y_device(size);
  auto* x_p{x_device.data()};
  auto* y_p{y_device.data()};

  void* args[] = {&x_p, &size, &y_p};
  cuda_module.LaunchKernel(0,
                           "cast_fp32_to_bf16_cuda_kernel",
                           blocks_per_grid,
                           threads_per_block,
                           args);
  CUDA_CALL(cudaDeviceSynchronize());

  std::vector<bfloat16> y_host = y_device.to_host();
  bool res = std::equal(x_host.begin(),
                        x_host.end(),
                        y_host.begin(),
                        [](float x, bfloat16 y) -> bool {
                          return std::abs(x - static_cast<float>(y)) < 1e-2f;
                        });
  PADDLE_ENFORCE_EQ(
      res,
      true,
      ::common::errors::PreconditionNotMet(
          "The difference between two arrays exceeds the bound."));
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
