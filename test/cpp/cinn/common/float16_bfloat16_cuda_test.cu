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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>
#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/float16.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace common {

#define CUDA_CALL(func)                                    \
  {                                                        \
    auto status = func;                                    \
    if (status != cudaSuccess) {                           \
      std::stringstream ss;                                \
      ss << "CUDA Error : " << cudaGetErrorString(status); \
      PADDLE_THROW(::common::errors::Fatal(ss.str()));     \
    }                                                      \
  }

class CudaMem {
 public:
  CudaMem() = default;

  void* mutable_data(size_t bytes) {
    PADDLE_ENFORCE_GT(
        bytes,
        0,
        ::common::errors::InvalidArgument("Cannot allocate empty memory!"));
    if (ptr) {
      PADDLE_ENFORCE_EQ(
          bytes,
          bytes_,
          ::common::errors::InvalidArgument("Try allocate memory twice!"));
      return ptr;
    }
    CUDA_CALL(cudaMalloc(&ptr, bytes));
    bytes_ = bytes;
    return ptr;
  }

  template <typename T>
  T* mutable_data(size_t num) {
    return reinterpret_cast<T*>(mutable_data(num * sizeof(T)));
  }

  void* data() const {
    PADDLE_ENFORCE_NOT_NULL(ptr,
                            ::common::errors::InvalidArgument(
                                "Pointer is null; please ensure it is properly "
                                "initialized before use."));
    return ptr;
  }

  template <typename T>
  T* data() const {
    return reinterpret_cast<T*>(data());
  }

  void MemcpyFromHost(const void* src,
                      size_t bytes,
                      cudaStream_t stream = nullptr) {
    PADDLE_ENFORCE_LE(
        bytes,
        bytes_,
        ::common::errors::InvalidArgument("Too many data need copy"));
    CUDA_CALL(cudaMemcpyAsync(ptr, src, bytes, cudaMemcpyHostToDevice, stream));
  }

  void MemcpyToHost(void* dst, size_t bytes, cudaStream_t stream = nullptr) {
    PADDLE_ENFORCE_LE(
        bytes,
        bytes_,
        ::common::errors::InvalidArgument("Too many data need copy"));
    CUDA_CALL(cudaMemcpyAsync(dst, ptr, bytes, cudaMemcpyDeviceToHost, stream));
  }

  ~CudaMem() {
    if (ptr) {
      cudaFree(ptr);
    }
    bytes_ = 0;
  }

 private:
  void* ptr{nullptr};
  size_t bytes_{0};
};

__global__ void cast_fp32_to_fp16_cuda_kernel(const float* input,
                                              const int num,
                                              float16* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    out[idx] = float16(input[idx]);
  }
}

__global__ void cast_fp16_to_fp32_cuda_kernel(const float16* input,
                                              const int num,
                                              float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    out[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void test_fp16_cuda_kernel(const float16* x,
                                      const float16* y,
                                      const int num,
                                      float16* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    float16 x_i = x[idx], y_i = y[idx];
    x_i += float16(1);

    out[idx] = (x_i + y_i) * (x_i - y_i);
  }
}

__global__ void cast_fp32_to_bf16_cuda_kernel(const float* input,
                                              const int num,
                                              bfloat16* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    out[idx] = bfloat16(input[idx]);
  }
}

__global__ void cast_bf16_to_fp32_cuda_kernel(const bfloat16* input,
                                              const int num,
                                              float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    out[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void test_bf16_cuda_kernel(const bfloat16* x,
                                      const bfloat16* y,
                                      const int num,
                                      bfloat16* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    bfloat16 x_i = x[idx], y_i = y[idx];
    x_i += bfloat16(1);

    out[idx] = (x_i + y_i) * (x_i - y_i);
  }
}

__global__ void test_fp32_cuda_kernel(const float* x,
                                      const float* y,
                                      const int num,
                                      float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    float x_i = x[idx], y_i = y[idx];
    x_i += 1.0f;

    out[idx] = (x_i + y_i) * (x_i - y_i);
  }
}

TEST(FP16_BF16, basic_cuda) {
#ifdef CUDA_VERSION
  LOG(INFO) << "CUDA version: " << CUDA_VERSION;
#endif

  int num = 2048;

  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreate(&stream));

  dim3 block = 1024;
  dim3 grid = (num + block.x - 1) / block.x;

  std::vector<float> x_fp32_host(num), y_fp32_host(num);
  {  // step1 : generate input data
    std::random_device r;
    std::default_random_engine eng(r());
    std::uniform_real_distribution<float> dis(1e-5f, 1.0f);

    for (int i = 0; i < num; ++i) {
      x_fp32_host[i] = dis(eng);
      y_fp32_host[i] = dis(eng);
    }
  }

  CudaMem x_fp32_device, y_fp32_device, out_fp32_device;
  {  // step2 : compute fp32 result
    auto x_fp32_ptr = x_fp32_device.mutable_data<float>(num);
    auto y_fp32_ptr = y_fp32_device.mutable_data<float>(num);
    auto out_fp32_ptr = out_fp32_device.mutable_data<float>(num);

    x_fp32_device.MemcpyFromHost(
        x_fp32_host.data(), num * sizeof(float), stream);
    y_fp32_device.MemcpyFromHost(
        y_fp32_host.data(), num * sizeof(float), stream);

    test_fp32_cuda_kernel<<<grid, block, 0, stream>>>(
        x_fp32_ptr, y_fp32_ptr, num, out_fp32_ptr);
  }

  CudaMem x_fp16_device, y_fp16_device, out_fp16_device;
  CudaMem x_bf16_device, y_bf16_device, out_bf16_device;
  {  // step3 : compute fp16/bf16 result
    // step3.1 : compute fp16 result
    auto x_fp16_ptr = x_fp16_device.mutable_data<float16>(num);
    auto y_fp16_ptr = y_fp16_device.mutable_data<float16>(num);
    auto out_fp16_ptr = out_fp16_device.mutable_data<float16>(num);

    cast_fp32_to_fp16_cuda_kernel<<<grid, block, 0, stream>>>(
        x_fp32_device.data<float>(), num, x_fp16_ptr);
    cast_fp32_to_fp16_cuda_kernel<<<grid, block, 0, stream>>>(
        y_fp32_device.data<float>(), num, y_fp16_ptr);

    test_fp16_cuda_kernel<<<grid, block, 0, stream>>>(
        x_fp16_ptr, y_fp16_ptr, num, out_fp16_ptr);

    // step3.2 : compute bf16 result
    auto x_bf16_ptr = x_bf16_device.mutable_data<bfloat16>(num);
    auto y_bf16_ptr = y_bf16_device.mutable_data<bfloat16>(num);
    auto out_bf16_ptr = out_bf16_device.mutable_data<bfloat16>(num);

    cast_fp32_to_bf16_cuda_kernel<<<grid, block, 0, stream>>>(
        x_fp32_device.data<float>(), num, x_bf16_ptr);
    cast_fp32_to_bf16_cuda_kernel<<<grid, block, 0, stream>>>(
        y_fp32_device.data<float>(), num, y_bf16_ptr);

    test_bf16_cuda_kernel<<<grid, block, 0, stream>>>(
        x_bf16_ptr, y_bf16_ptr, num, out_bf16_ptr);
  }

  CudaMem fp32res_fp16_device;
  CudaMem fp32res_bf16_device;
  {  // step4 : cast fp16/bf16 result to fp32 result
    // step4.1 : cast fp16 result to fp32 result
    auto fp32res_fp16_ptr = fp32res_fp16_device.mutable_data<float>(num);
    cast_fp16_to_fp32_cuda_kernel<<<grid, block, 0, stream>>>(
        out_fp16_device.data<float16>(), num, fp32res_fp16_ptr);

    // step4.2 : cast bf16 result to fp32 result
    auto fp32res_bf16_ptr = fp32res_bf16_device.mutable_data<float>(num);
    cast_bf16_to_fp32_cuda_kernel<<<grid, block, 0, stream>>>(
        out_bf16_device.data<bfloat16>(), num, fp32res_bf16_ptr);
  }

  std::vector<float> out_fp32_host(num), out_fp16_host(num), out_bf16_host(num);
  {  // step5 : copy result from device to host
    out_fp32_device.MemcpyToHost(
        out_fp32_host.data(), num * sizeof(float), stream);
    fp32res_fp16_device.MemcpyToHost(
        out_fp16_host.data(), num * sizeof(float), stream);
    fp32res_bf16_device.MemcpyToHost(
        out_bf16_host.data(), num * sizeof(float), stream);
  }

  CUDA_CALL(cudaStreamSynchronize(stream));

  for (int i = 0; i < num; ++i) {
    ASSERT_NEAR(out_fp32_host[i], out_fp16_host[i], 1e-2f);
    ASSERT_NEAR(out_fp32_host[i], out_bf16_host[i], 1e-1f);
  }

  CUDA_CALL(cudaStreamDestroy(stream));
}

}  // namespace common
}  // namespace cinn
