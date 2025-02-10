// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/fusion/cutlass/gemm_epilogue/gemm_epilogue_util.h"
#include <cmath>
#include <iostream>

namespace phi {
namespace fusion {
namespace cutlass_internal {

template <typename T>
float diff(const T *C_cutlass, const T *C_naive, size_t n) {
  float max_diff = -1.;
  for (size_t i = 0; i < n; i++) {
    float cutlass_value = static_cast<float>(C_cutlass[i]);
    float naive_value = static_cast<float>(C_naive[i]);
    if (std::abs(naive_value - cutlass_value) > max_diff) {
      max_diff = std::abs(naive_value - cutlass_value);
    }
  }
  return max_diff;
}

__device__ inline float naive_tanh(float x) {
  if (x > 0)
    return (1 - exp(-2 * x)) / (1 + exp(-2 * x));
  else
    return (exp(2 * x) - 1) / (1 + exp(2 * x));
}

template <typename T = half>
__global__ void naive_gemm_epilogue_kernel(const T *input,
                                           const T *weight,
                                           const T *bias,
                                           T *output,
                                           size_t M,
                                           size_t N,
                                           size_t K,
                                           size_t lda,
                                           size_t ldb,
                                           size_t ldd,
                                           float leaky_alpha,
                                           bool isVec_bias,
                                           OpType op_type) {
  size_t j = threadIdx.x + blockIdx.x * blockDim.x;
  size_t i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0.;
    for (size_t k = 0; k < K; ++k) {
      float input_ele = static_cast<float>(input[i * lda + k]);
      float weight_ele = static_cast<float>(weight[k * ldb + j]);
      accumulator += input_ele * weight_ele;
    }
    if (isVec_bias) {
      accumulator += static_cast<float>(bias[j]);
    } else {
      accumulator += static_cast<float>(bias[i * ldd + j]);
    }

    switch (op_type) {
      case MATMUL_ADD:
        break;
      case MATMUL_ADD_RELU:
        accumulator = accumulator > 0 ? accumulator : 0;
        break;
      case MATMUL_ADD_GELU:
        accumulator =
            0.5 * accumulator *
            (1 +
             naive_tanh(std::sqrt(2 / M_PI) *
                        (accumulator + 0.044715 * std::pow(accumulator, 3))));
        break;
      // case MATMUL_ADD_LEAKY_RELU:
      //   accumulator = accumulator > 0 ? accumulator : (accumulator *
      //   leaky_alpha); break;
      // case MATMUL_ADD_SIGMOID:
      //   accumulator = 1.f / (1.f + std::exp(-accumulator));
      //   break;
      // case MATMUL_ADD_SILU:
      //   accumulator = accumulator * (1.f / (1 + exp(-accumulator)));
      //   break;
      default:
        break;
    }
    output[i * ldd + j] = (T)accumulator;
  }
}

template <typename T>
float gemm_epilogue_diff_gpu(const GemmEpilogueAllParams &params,
                             OpType op_type) {
  const T *input = reinterpret_cast<const T *>(params.input);
  const T *weight = reinterpret_cast<const T *>(params.weight);
  const T *bias = reinterpret_cast<const T *>(params.bias);
  T *output_cutlass_D = reinterpret_cast<T *>(params.output);
  size_t M = params.m, N = params.n, K = params.k;
  size_t lda = params.lda, ldb = params.ldb, ldd = params.ldd;
  float leaky_alpha = params.leaky_alpha;
  bool isVec_bias = params.isVec_bias;

  size_t outSize = sizeof(T) * M * N;
  T *output_naive_D;
  CUDA_CHECK(cudaMalloc(&output_naive_D, outSize));
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  naive_gemm_epilogue_kernel<T><<<grid, block>>>(input,
                                                 weight,
                                                 bias,
                                                 output_naive_D,
                                                 M,
                                                 N,
                                                 K,
                                                 lda,
                                                 ldb,
                                                 ldd,
                                                 leaky_alpha,
                                                 isVec_bias,
                                                 op_type);
  cudaGetLastError();
  CUDA_CHECK(cudaDeviceSynchronize());

  T *output_cutlass_H = reinterpret_cast<T *>(malloc(outSize));
  CUDA_CHECK(cudaMemcpy(
      output_cutlass_H, output_cutlass_D, outSize, cudaMemcpyDeviceToHost));
  T *output_naive_H = reinterpret_cast<T *>(malloc(outSize));
  CUDA_CHECK(cudaMemcpy(
      output_naive_H, output_naive_D, outSize, cudaMemcpyDeviceToHost));

  float max_diff = diff(output_cutlass_H, output_naive_H, M * N);

  free(output_cutlass_H);
  free(output_naive_H);
  cudaFree(output_naive_D);
  return max_diff;
}

std::string OpType2String(OpType op_type) {
  switch (op_type) {
    case MATMUL_ADD:
      return "matmul_add";
    case MATMUL_ADD_RELU:
      return "matmul_add_relu";
    case MATMUL_ADD_GELU:
      return "matmul_add_gelu";
    // case MATMUL_ADD_SIGMOID:
    //   return "matmul_add_sigmoid";
    // case MATMUL_ADD_LEAKY_RELU:
    //   return "matmul_add_leaky_relu";
    // case MATMUL_ADD_SILU:
    //   return "matmul_add_silu";
    default:
      break;
  }
  return "unnamed_op";
}

int ProfileToGetBestConfig(
    const std::vector<std::function<cutlass::Status(GemmEpilogueAllParams)>>
        &all_func,
    const GemmEpilogueAllParams &params,
    OpType op_type) {
  std::cout << "we are running for problem: [" << params.m << ", " << params.n
            << ", " << params.k << "]" << std::endl;

  constexpr int WARMUP = 10;
  constexpr int REPEAT = 10;
  float min_time = 100000.f;
  int min_time_index = -1;
  for (int i = 0; i < all_func.size(); i++) {
    cutlass::Status status;
    auto func = all_func[i];
    // sizeof(half) attention！！
    CUDA_CHECK(
        cudaMemset(params.output, 0, sizeof(half) * params.m * params.n));
    status = func(params);
    if (status != cutlass::Status::kSuccess) continue;

    for (int ii = 0; ii < WARMUP; ii++) {
      status = func(params);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    for (int ii = 0; ii < REPEAT; ii++) {
      status = func(params);
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, beg, end));

    if (elapsed_time < min_time && status == cutlass::Status::kSuccess) {
      min_time = elapsed_time;
      min_time_index = i;

      if (params.data_type == GemmEpilogueDataType::fp16) {
        // debug code
        std::cout << "fp16_" << OpType2String(op_type) << ": tactic " << i
                  << " has max diff "
                  << gemm_epilogue_diff_gpu<half>(params, op_type)
                  << " compared with baseline,"
                  << "cost_time: " << elapsed_time << "ms." << std::endl;
      } else if (params.data_type == GemmEpilogueDataType::bf16) {
        // debug code
        std::cout << "bf16_" << OpType2String(op_type) << ": tactic " << i
                  << " has max diff "
                  << gemm_epilogue_diff_gpu<__nv_bfloat16>(params, op_type)
                  << " compared with baseline,"
                  << "cost_time: " << elapsed_time << "ms." << std::endl;
      } else if (params.data_type == GemmEpilogueDataType::fp32) {
        // debug code
        std::cout << "fp32_" << OpType2String(op_type) << ": tactic " << i
                  << " has max diff "
                  << gemm_epilogue_diff_gpu<float>(params, op_type)
                  << " compared with baseline,"
                  << "cost_time: " << elapsed_time << "ms." << std::endl;
      }
    }
  }

  if (min_time_index < 0) {
    std::cout << "Can't find any cutlass config for " << OpType2String(op_type)
              << std::endl;
  }
  return min_time_index;
}

template float gemm_epilogue_diff_gpu<float>(
    const GemmEpilogueAllParams &params, OpType op_type);
template float gemm_epilogue_diff_gpu<half>(const GemmEpilogueAllParams &params,
                                            OpType op_type);
template float gemm_epilogue_diff_gpu<__nv_bfloat16>(
    const GemmEpilogueAllParams &params, OpType op_type);

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
