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

#include "paddle/cinn/runtime/cuda/cuda_util.h"

#include <absl/container/flat_hash_map.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <string>
#ifdef CINN_WITH_CUDNN
#include <cudnn.h>
#endif

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/runtime/cuda/cublas_util.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/cinn/utils/timer.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace runtime {
namespace cuda {

class CublasHandle {
 public:
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;
  ~CublasHandle() {
    CUBLAS_CALL(cublasDestroy(cuhandle));
    CUDA_CALL(cudaStreamDestroy(custream));
  }
  static CublasHandle &GetInstance() {
    static CublasHandle instance;
    return instance;
  }
  cudaStream_t GetCuStream() { return custream; }
  cublasHandle_t &GetCublasHandle() { return cuhandle; }

 private:
  CublasHandle() {
    CUDA_CALL(cudaStreamCreate(&custream));
    CUBLAS_CALL(cublasCreate(&cuhandle));
    cudaMemPool_t mem_pool;
    CUDA_CALL(cudaDeviceGetMemPool(&mem_pool, 0));

    uint64_t threshold = UINT32_MAX;
    CUDA_CALL(cudaMemPoolSetAttribute(
        mem_pool, cudaMemPoolAttrReleaseThreshold, &threshold));

    int enable = 1;
    CUDA_CALL(cudaMemPoolSetAttribute(
        mem_pool, cudaMemPoolReuseFollowEventDependencies, &enable));
    CUDA_CALL(cudaMemPoolSetAttribute(
        mem_pool, cudaMemPoolReuseAllowInternalDependencies, &enable));
  }
  cudaStream_t custream;
  cublasHandle_t cuhandle;
};

int64_t cinn_get_value_in_cuda_kernel_args(void *v_args, int idx) {
  KernelArgsNode *args = static_cast<KernelArgsNode *>(v_args);
  return args[idx].symbol_val;
}

void *cinn_get_item_in_cuda_kernel_args(void *v_args, int idx) {
  KernelArgsNode *args = static_cast<KernelArgsNode *>(v_args);
  return static_cast<void *>(&(args[idx].data_ptr));
}

void cinn_call_cuda_kernel(void *kernel_fn,
                           void *v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           int shared_memory_bytes,
                           void *stream) {
  VLOG(3) << "cinn_call_cuda_kernel, grid_dim={" << grid_x << ", " << grid_y
          << ", " << grid_z << "}, block_dim={" << block_x << ", " << block_y
          << ", " << block_z << "}, num_args=" << num_args
          << ", shared_memory_bytes=" << shared_memory_bytes
          << ", stream=" << stream << ", kernel_fn=" << kernel_fn;
  std::vector<void *> kernel_args;
  {
    kernel_args.reserve(num_args);
    KernelArgsNode *args = static_cast<KernelArgsNode *>(v_args);
    for (int idx = 0; idx < num_args; ++idx) {
      // kernel_args.emplace_back( &(args[idx]));  // NOLINT

      if (args[idx].args_type == KernelArgsType::void_p_args_type) {
        kernel_args.emplace_back(&(args[idx].data_ptr));  // NOLINT
      } else if (args[idx].args_type == KernelArgsType::int64_args_type) {
        kernel_args.emplace_back(&(args[idx].symbol_val));  // NOLINT
      } else {
        PADDLE_THROW(
            phi::errors::PreconditionNotMet("Only support void* and int64"));
      }
    }
  }

  {
    cinn::utils::RecordEvent record_run("cuLaunchKernel",
                                        cinn::utils::EventType::kInstruction);
    CUDA_DRIVER_CALL(cuLaunchKernel(static_cast<CUfunction>(kernel_fn),
                                    grid_x,
                                    grid_y,
                                    grid_z,
                                    block_x,
                                    block_y,
                                    block_z,
                                    shared_memory_bytes,
                                    static_cast<CUstream>(stream),
                                    kernel_args.data(),
                                    nullptr))
  }
}

void cinn_call_cublas(void *v_args,
                      int num_args,
                      bool trans_a,
                      bool trans_b,
                      bool trans_o,
                      float alpha,
                      float beta,
                      int a1,
                      int a2,
                      int a3,
                      int a4,
                      int b1,
                      int b2,
                      int b3,
                      int b4,
                      void *stream) {
  cinn::utils::RecordEvent record_run("cinn_call_cublas",
                                      cinn::utils::EventType::kInstruction);
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      ::common::errors::InvalidArgument(
          "Expected number of arguments is 3, but received %d.", num_args));
  cublasHandle_t &cuhandle = CublasHandle::GetInstance().GetCublasHandle();
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(cuhandle, custream));
  VLOG(3) << "a1 ~ a4: " << a1 << " " << a2 << " " << a3 << " " << a4;
  VLOG(3) << "b1 ~ b4: " << b1 << " " << b2 << " " << b3 << " " << b4;
  VLOG(3) << "trans_a: " << trans_a << ", trans_b: " << trans_b
          << ", trans_o: " << trans_o;

  void *A = args[0].operator cinn_buffer_t *()->memory;
  void *B = args[1].operator cinn_buffer_t *()->memory;
  void *C = args[2].operator cinn_buffer_t *()->memory;

  int m = trans_o ? (trans_a ? a4 : a3) : (trans_b ? b3 : b4);
  int n = trans_o ? (trans_b ? b3 : b4) : (trans_a ? a4 : a3);
  int k = trans_a ? a3 : a4;

  VLOG(3) << "m: " << m << ", n: " << n << ", k: " << k;

  cublasOperation_t trans_op_l = trans_o
                                     ? (trans_a ? CUBLAS_OP_N : CUBLAS_OP_T)
                                     : (trans_b ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasOperation_t trans_op_r = trans_o
                                     ? (trans_b ? CUBLAS_OP_N : CUBLAS_OP_T)
                                     : (trans_a ? CUBLAS_OP_T : CUBLAS_OP_N);
  int ldl = trans_op_l == CUBLAS_OP_N
                ? m
                : k;  // trans_o ? (trans_a ? k : m) : (trans_b ? k : m);
  int ldr = trans_op_r == CUBLAS_OP_N
                ? k
                : n;  // trans_o ? (trans_b ? n : k) : (trans_a ? n : k);
  int ldc = m;

  void *lhs = trans_o ? A : B;
  void *rhs = trans_o ? B : A;

  cudaDataType_t cuda_dtype;
  auto type_code = args[0].operator cinn_buffer_t *()->type.code;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  int bytes = args[0].operator cinn_buffer_t *()->type.bits / CHAR_BIT;
  if (is_float && bytes == sizeof(cinn::common::float16)) {
    cuda_dtype = CUDA_R_16F;
  } else if (is_float && bytes == sizeof(float)) {
    cuda_dtype = CUDA_R_32F;
  } else if (is_float && bytes == sizeof(double)) {
    cuda_dtype = CUDA_R_64F;
  } else if (is_bfloat16) {
    cuda_dtype = CUDA_R_16BF;
  } else {
    std::stringstream ss;
    ss << "unsupported cublas data type: " << static_cast<int>(type_code)
       << ", bytes = " << bytes;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }

  if (a1 * a2 * b1 * b2 == 1) {
    VLOG(3) << "call cublasGemm for a1 * a2 * b1 * b2 == 1";
    cinn::utils::RecordEvent record_run("Call cublasGemm",
                                        cinn::utils::EventType::kInstruction);
    CUBLAS_CALL(cublasGemm(cuda_dtype,
                           cuhandle,
                           trans_op_l,
                           trans_op_r,
                           m,
                           n,
                           k,
                           alpha,
                           lhs,
                           ldl,
                           rhs,
                           ldr,
                           beta,
                           C,
                           ldc));
  } else if (a1 * b1 == 1) {
    CHECK(a2 == b2 || a2 == 1 || b2 == 1);
    if (b2 == 1 && trans_op_r == CUBLAS_OP_N) {
      // In case of [1, bs, M, K] * [1, 1, K, N]
      VLOG(3) << "call cublasGemm for a1 * b1 = 1, b2 = 1, trans_op_r:"
              << trans_op_r;
      cinn::utils::RecordEvent record_run("Call cublasGemm",
                                          cinn::utils::EventType::kInstruction);
      CUBLAS_CALL(cublasGemm(cuda_dtype,
                             cuhandle,
                             trans_op_l,
                             trans_op_r,
                             m,
                             a2 * n,
                             k,
                             alpha,
                             lhs,
                             ldl,
                             A,
                             ldr,
                             beta,
                             C,
                             ldc));
    } else {
      int stride_l = trans_o ? (a2 > 1 ? a3 * a4 : 0) : (b2 > 1 ? b3 * b4 : 0);
      int stride_r = trans_o ? (b2 > 1 ? b3 * b4 : 0) : (a2 > 1 ? a3 * a4 : 0);
      int batch = std::max(a2, b2);
      VLOG(3) << "call cublasGemmStridedBatched with a1*b1 = 1, stride_l = "
              << stride_l << ", stride_r = " << stride_r
              << ", batch = " << batch << ", dtype = " << cuda_dtype;
      cinn::utils::RecordEvent record_run("Call cublasGemmStridedBatched",
                                          cinn::utils::EventType::kInstruction);
      CUBLAS_CALL(cublasGemmStridedBatched(cuda_dtype,
                                           cuhandle,
                                           trans_op_l,
                                           trans_op_r,
                                           m,
                                           n,
                                           k,
                                           alpha,
                                           lhs,
                                           ldl,
                                           stride_l,
                                           rhs,
                                           ldr,
                                           stride_r,
                                           beta,
                                           C,
                                           ldc,
                                           m * n,
                                           batch));
    }
  } else {
    int l1 = trans_o ? a1 : b1, l2 = trans_o ? a2 : b2, l3 = trans_o ? a3 : b3,
        l4 = trans_o ? a4 : b4;
    int r1 = trans_o ? b1 : a1, r2 = trans_o ? b2 : a2, r3 = trans_o ? b3 : a3,
        r4 = trans_o ? b4 : a4;

    if ((l1 == r1 && l2 == r2) || (l1 == 1 && l2 == 1) ||
        (r1 == 1 && r2 == 1)) {
      int stride_l = (l1 == 1 && l2 == 1) ? 0 : l3 * l4;
      int stride_r = (r1 == 1 && r2 == 1) ? 0 : r3 * r4;

      // four types matmul:
      // (N, L) * (N, L) , (N, 1) * (N, 1)
      // (N, L) * (1, 1) , (1, 1) * (N, L)
      VLOG(3) << "call cublasGemmStridedBatched for stride_l = " << stride_l
              << ", stride_r = " << stride_r
              << ", batch = " << std::max(l1, r1) * std::max(l2, r2);
      cinn::utils::RecordEvent record_run("Call cublasGemmStridedBatched",
                                          cinn::utils::EventType::kInstruction);
      CUBLAS_CALL(
          cublasGemmStridedBatched(cuda_dtype,
                                   cuhandle,
                                   trans_op_l,
                                   trans_op_r,
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   lhs,
                                   ldl,
                                   stride_l,
                                   rhs,
                                   ldr,
                                   stride_r,
                                   beta,
                                   C,
                                   ldc,
                                   m * n,
                                   std::max(l1, r1) * std::max(l2, r2)));
    } else {
      cinn::utils::RecordEvent record_run("Call cublasGemmBatched",
                                          cinn::utils::EventType::kInstruction);
      // (N, L) / (N, 1) / (1, L)
      int bstride_l =
          (l1 != 1 && l2 != 1) ? (l2 * m * k) : ((l1 != 1) ? m * k : 0);
      // (N, L) / (N, 1) / (1, L)
      int bstride_r =
          (r1 != 1 && r2 != 1) ? (r2 * k * n) : ((r1 != 1) ? k * n : 0);
      int bstride_c = std::max(l2, r2) * m * n;

      int stride_l = l2 == 1 ? 0 : l3 * l4;
      int stride_r = r2 == 1 ? 0 : r3 * r4;
      // six type matmul:
      // (N, L) * (N, 1) , (N, L) * (1, L)
      // (N, 1) * (N, L) , (1, L) * (N, L)
      // (N, 1) * (1, L) , (1, L) * (N, 1)

      void **ptr_arr = nullptr;
      cudaStream_t g_stream = CublasHandle::GetInstance().GetCuStream();
      CUDA_CALL(cudaMallocAsync(
          &ptr_arr,
          sizeof(void *) * 3 * std::max(l1, r1) * std::max(l2, r2),
          g_stream));

      std::vector<void *> ptr(3 * std::max(l1, r1) * std::max(l2, r2));
      void **ptr_a = ptr.data();
      void **ptr_b = ptr.data() + std::max(l1, r1) * std::max(l2, r2);
      void **ptr_c = ptr.data() + std::max(l1, r1) * std::max(l2, r2) * 2;

      for (int idx = 0, index = 0; idx < std::max(l1, r1); ++idx) {
        for (int idy = 0; idy < std::max(l2, r2); ++idy) {
          ptr_a[index] = reinterpret_cast<uint8_t *>(lhs) +
                         (idx * bstride_l + idy * stride_l) * bytes;
          ptr_b[index] = reinterpret_cast<uint8_t *>(rhs) +
                         (idx * bstride_r + idy * stride_r) * bytes;
          ptr_c[index] = reinterpret_cast<uint8_t *>(C) +
                         (idx * bstride_c + idy * m * n) * bytes;
          ++index;
        }
      }
      CUDA_CALL(cudaMemcpyAsync(ptr_arr,
                                ptr.data(),
                                ptr.size() * sizeof(void *),
                                cudaMemcpyHostToDevice,
                                g_stream));
      CUDA_CALL(cudaStreamSynchronize(g_stream));

      CUBLAS_CALL(
          cublasGemmBatched(cuda_dtype,
                            cuhandle,
                            trans_op_l,
                            trans_op_r,
                            m,
                            n,
                            k,
                            alpha,
                            ptr_arr,
                            ldl,
                            ptr_arr + std::max(l1, r1) * std::max(l2, r2),
                            ldr,
                            beta,
                            ptr_arr + std::max(l1, r1) * std::max(l2, r2) * 2,
                            ldc,
                            std::max(l1, r1) * std::max(l2, r2)));
      CUDA_CALL(cudaFreeAsync(ptr_arr, custream));
    }
  }
}

void cinn_call_batched_cublas(void *v_args,
                              int num_args,
                              int opside,
                              bool trans_a,
                              bool trans_b,
                              bool trans_o,
                              float alpha,
                              float beta,
                              int a1,
                              int a2,
                              int a3,
                              int a4,
                              int b1,
                              int b2,
                              int b3,
                              int b4,
                              void *stream) {
  // A * [B, C, D, ...] or [B, C, D, ...] * A
  PADDLE_ENFORCE_EQ((num_args - 1) % 2,
                    0,
                    ::common::errors::PreconditionNotMet(
                        "(num_args - 1) should be divided by 2."));
  cublasHandle_t &cuhandle = CublasHandle::GetInstance().GetCublasHandle();
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(cuhandle, custream));

  cudaDataType_t cuda_dtype;
  auto type_code = args[0].operator cinn_buffer_t *()->type.code;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  int bytes = args[0].operator cinn_buffer_t *()->type.bits / CHAR_BIT;
  if (is_float && bytes == sizeof(cinn::common::float16)) {
    cuda_dtype = CUDA_R_16F;
  } else if (is_float && bytes == sizeof(float)) {
    cuda_dtype = CUDA_R_32F;
  } else if (is_float && bytes == sizeof(double)) {
    cuda_dtype = CUDA_R_64F;
  } else if (is_bfloat16) {
    cuda_dtype = CUDA_R_16BF;
  } else {
    std::stringstream ss;
    ss << "unsupported cublas data type: " << static_cast<int>(type_code)
       << ", bytes = " << bytes;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }

  int m = trans_o ? (trans_a ? a4 : a3) : (trans_b ? b3 : b4);
  int n = trans_o ? (trans_b ? b3 : b4) : (trans_a ? a4 : a3);
  int k = trans_a ? a3 : a4;

  cublasOperation_t trans_op_l = trans_o
                                     ? (trans_a ? CUBLAS_OP_N : CUBLAS_OP_T)
                                     : (trans_b ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasOperation_t trans_op_r = trans_o
                                     ? (trans_b ? CUBLAS_OP_N : CUBLAS_OP_T)
                                     : (trans_a ? CUBLAS_OP_T : CUBLAS_OP_N);
  int ldl = trans_op_l == CUBLAS_OP_N
                ? m
                : k;  // trans_o ? (trans_a ? k : m) : (trans_b ? k : m);
  int ldr = trans_op_r == CUBLAS_OP_N
                ? k
                : n;  // trans_o ? (trans_b ? n : k) : (trans_a ? n : k);
  int ldc = m;

  int l1 = trans_o ? a1 : b1, l2 = trans_o ? a2 : b2, l3 = trans_o ? a3 : b3,
      l4 = trans_o ? a4 : b4;
  int r1 = trans_o ? b1 : a1, r2 = trans_o ? b2 : a2, r3 = trans_o ? b3 : a3,
      r4 = trans_o ? b4 : a4;

  // (N, L): L * M * K
  // (N, 1): 1 * M * K
  // (1, L): 0
  // (1, 1): 0
  int bstride_l = (l1 != 1 && l2 != 1) ? (l2 * m * k) : ((l1 != 1) ? m * k : 0);
  int bstride_r = (r1 != 1 && r2 != 1) ? (r2 * k * n) : ((r1 != 1) ? k * n : 0);
  int bstride_c = std::max(l2, r2) * m * n;
  // (N, L): K * N
  // (N, 1): 0
  // (1, L): K * N
  // (1, 1): 0
  int stride_l = l2 == 1 ? 0 : l3 * l4;
  int stride_r = r2 == 1 ? 0 : r3 * r4;

  int num_gemm = ((num_args - 1) / 2);
  std::vector<void *> ptr(3 * std::max(l1, r1) * std::max(l2, r2) * num_gemm);
  void **ptr_a = ptr.data();
  void **ptr_b = ptr.data() + std::max(l1, r1) * std::max(l2, r2) * num_gemm;
  void **ptr_c =
      ptr.data() + std::max(l1, r1) * std::max(l2, r2) * num_gemm * 2;

  void **ptr_arr = nullptr;
  cudaStream_t g_stream = CublasHandle::GetInstance().GetCuStream();
  CUDA_CALL(cudaMallocAsync(&ptr_arr, sizeof(void *) * ptr.size(), g_stream));

  for (int g = 0, index = 0; g < num_gemm; ++g) {
    void *A = args[0].operator cinn_buffer_t *()->memory;
    void *B = args[1 + g].operator cinn_buffer_t *()->memory;
    void *C = args[1 + num_gemm + g].operator cinn_buffer_t *()->memory;

    // if opside is 1, exchange A,B.
    if (opside) {
      auto tmp = A;
      A = B;
      B = tmp;
    }

    void *lhs = trans_o ? A : B;
    void *rhs = trans_o ? B : A;

    for (int idx = 0; idx < std::max(l1, r1); ++idx) {
      for (int idy = 0; idy < std::max(l2, r2); ++idy) {
        ptr_a[index] = reinterpret_cast<uint8_t *>(lhs) +
                       (idx * bstride_l + idy * stride_l) * bytes;
        ptr_b[index] = reinterpret_cast<uint8_t *>(rhs) +
                       (idx * bstride_r + idy * stride_r) * bytes;
        ptr_c[index] = reinterpret_cast<uint8_t *>(C) +
                       (idx * bstride_c + idy * m * n) * bytes;
        ++index;
      }
    }
  }

  CUDA_CALL(cudaMemcpyAsync(ptr_arr,
                            ptr.data(),
                            ptr.size() * sizeof(void *),
                            cudaMemcpyHostToDevice,
                            g_stream));
  CUDA_CALL(cudaStreamSynchronize(g_stream));

  CUBLAS_CALL(cublasGemmBatched(
      cuda_dtype,
      cuhandle,
      trans_op_l,
      trans_op_r,
      m,
      n,
      k,
      alpha,
      ptr_arr,
      ldl,
      ptr_arr + std::max(l1, r1) * std::max(l2, r2) * num_gemm,
      ldr,
      beta,
      ptr_arr + std::max(l1, r1) * std::max(l2, r2) * 2 * num_gemm,
      ldc,
      std::max(l1, r1) * std::max(l2, r2) * num_gemm));
  CUDA_CALL(cudaFreeAsync(ptr_arr, custream));
}

void cinn_call_cuda_memset(
    void *v_args, int num_args, int value, size_t count, void *stream) {
  PADDLE_ENFORCE_EQ(num_args,
                    1,
                    ::common::errors::PreconditionNotMet(
                        "The cinn_call_cuda_memset only accept a output."));
  VLOG(4) << "call cinn_call_cuda_memset with value=" << value
          << ", count=" << count;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *output = args[0].operator cinn_buffer_t *()->memory;

  cudaStream_t custream = static_cast<cudaStream_t>(stream);

  CUDA_CALL(cudaMemsetAsync(output, value, count, custream));
}

void cinn_call_cuda_memcpy(void *v_args,
                           int num_args,
                           size_t count,
                           void *stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      2,
      ::common::errors::PreconditionNotMet(
          "The cinn_call_cuda_memset only accept a input and a output."));
  VLOG(4) << "call cinn_call_cuda_memcpy with count=" << count;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *input = args[0].operator cinn_buffer_t *()->memory;
  void *output = args[1].operator cinn_buffer_t *()->memory;

  cudaStream_t custream = static_cast<cudaStream_t>(stream);

  CUDA_CALL(cudaMemcpyAsync(
      output, input, count, cudaMemcpyDeviceToDevice, custream));
}

#ifdef CINN_WITH_CUDNN
class CudnnHandle {
 public:
  CudnnHandle(const CudnnHandle &) = delete;
  CudnnHandle &operator=(const CudnnHandle &) = delete;
  ~CudnnHandle() {
    CUDNN_CALL(cudnnDestroy(cuhandle_));
    if (workspace_) {
      CUDA_CALL(cudaFree(workspace_));
    }
  }
  static CudnnHandle &GetInstance() {
    static CudnnHandle instance;
    return instance;
  }
  cudnnHandle_t &GetCudnnHandle() { return cuhandle_; }
  void *GetWorkSpace(size_t size) {
    if (size_ >= size) {
      return workspace_;
    } else {
      if (workspace_) {
        CUDA_CALL(cudaFree(workspace_));
      }
      size_ = size;
      CUDA_CALL(cudaMalloc(&workspace_, size_));
      return workspace_;
    }
  }

 private:
  CudnnHandle() : workspace_(nullptr), size_(0) {
    CUDNN_CALL(cudnnCreate(&cuhandle_));
  }
  cudnnHandle_t cuhandle_;
  void *workspace_;
  size_t size_;
};

class ConvAlgoMap {
 public:
  ConvAlgoMap(const ConvAlgoMap &) = delete;
  ConvAlgoMap &operator=(const ConvAlgoMap &) = delete;
  static ConvAlgoMap &GetInstance() {
    static ConvAlgoMap instance;
    return instance;
  }
  void InsertAlgo(const std::string &key, const int algo) {
    algo_map_[key] = algo;
  }
  int GetAlgo(const std::string &key) {
    return algo_map_.count(key) ? algo_map_[key] : -1;
  }

 private:
  ConvAlgoMap() {}
  absl::flat_hash_map<std::string, int> algo_map_;
};

cudnnDataType_t convert_to_cudnn_dtype(void *v_args, int num_args) {
  PADDLE_ENFORCE_GT(num_args,
                    0,
                    ::common::errors::PreconditionNotMet(
                        "the number of arguments must larger than zero"));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  auto type_code = args[0].operator cinn_buffer_t *()->type.code;
  int bits = args[0].operator cinn_buffer_t *()->type.bits;
  for (int i = 1; i < num_args; ++i) {
    auto t = args[i].operator cinn_buffer_t *()->type.code;
    int b = args[0].operator cinn_buffer_t *()->type.bits;
    if (t != type_code || bits != b) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "The types of all arguments need to be consistent."));
    }
  }
  cudnnDataType_t data_type;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  if (is_float && bits == 16) {
    data_type = CUDNN_DATA_HALF;
  } else if (is_float && bits == 32) {
    data_type = CUDNN_DATA_FLOAT;
  } else if (is_bfloat16) {
    data_type = CUDNN_DATA_BFLOAT16;
  } else if (is_float && bits == 64) {
    data_type = CUDNN_DATA_DOUBLE;
  } else {
    std::stringstream ss;
    ss << "unsupported cudnn data type: " << static_cast<int>(type_code)
       << ", bits = " << bits;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  return data_type;
}

cudnnDataType_t get_cudnn_compute_dtype(cudnnDataType_t data_type) {
  switch (data_type) {
    case CUDNN_DATA_FLOAT:
    case CUDNN_DATA_HALF:
    case CUDNN_DATA_BFLOAT16:
      return CUDNN_DATA_FLOAT;
    case CUDNN_DATA_DOUBLE:
      return CUDNN_DATA_DOUBLE;
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "unsupported cudnn data type, only support "
          "float16/bfloat16/float32/float64 now!"));
  }
  return CUDNN_DATA_FLOAT;
}

std::string debug_cudnn_tensor_format(cudnnTensorFormat_t tensor_format) {
  switch (tensor_format) {
    case CUDNN_TENSOR_NCHW:
      return "NCHW";
    case CUDNN_TENSOR_NHWC:
      return "NHWC";
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Only support NCHW and NHWC data layout\n"));
  }
  return "";
}

std::string debug_cudnn_tensor_dtype(cudnnDataType_t tensor_dtype) {
  switch (tensor_dtype) {
    case CUDNN_DATA_FLOAT:
      return "float32";
    case CUDNN_DATA_HALF:
      return "float16";
    case CUDNN_DATA_BFLOAT16:
      return "bfloat16";
    case CUDNN_DATA_DOUBLE:
      return "float64";
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Only support float16/bfloat16/float32/float64 now!"));
  }
  return "";
}

std::string debug_cudnn_pool_mode(cudnnPoolingMode_t pool_mode) {
  switch (pool_mode) {
    case CUDNN_POOLING_MAX:
      return "max";
    case CUDNN_POOLING_MAX_DETERMINISTIC:
      return "max_deterministic";
    case CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
      return "avg_include_padding";
    case CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
      return "avg_exclude_padding";
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Pool only support max and avg now!"));
  }
  return "";
}

void cinn_call_cudnn_conv2d_forward(void *v_args,
                                    int num_args,
                                    int format,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int filter_n,
                                    int filter_c,
                                    int filter_h,
                                    int filter_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation_h,
                                    int dilation_w,
                                    int groups,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    void *stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 3, but received %d.", num_args));
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_w = args[1].operator cinn_buffer_t *()->memory;
  void *_y = args[2].operator cinn_buffer_t *()->memory;

  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);
  cudnnDataType_t data_type = convert_to_cudnn_dtype(v_args, num_args);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc,
                                        data_type,
                                        tensor_format,
                                        filter_n,
                                        filter_c,
                                        filter_h,
                                        filter_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(
      cudnnSetConvolution2dDescriptor(conv_desc,
                                      pad_h,
                                      pad_w,
                                      stride_h,
                                      stride_w,
                                      dilation_h,
                                      dilation_w,
                                      CUDNN_CROSS_CORRELATION,
                                      get_cudnn_compute_dtype(data_type)));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
                                        tensor_format,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  auto &conv_algo_map = ConvAlgoMap::GetInstance();
  std::string hash_key =
      "conv2d forward, layout=" + debug_cudnn_tensor_format(tensor_format) +
      ", dtype=" + debug_cudnn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(filter_n) + "," +
      std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
      std::to_string(filter_w) + "}, output_nchw={" + std::to_string(output_n) +
      "," + std::to_string(output_c) + "," + std::to_string(output_h) + "," +
      std::to_string(output_w) + "}";
  VLOG(4) << hash_key;
  cudnnConvolutionFwdAlgo_t algo;
  int algo_int = conv_algo_map.GetAlgo(hash_key);
  if (algo_int >= 0) {
    algo = cudnnConvolutionFwdAlgo_t(algo_int);
  } else {
    int count = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
        handle, x_desc, w_desc, conv_desc, y_desc, 1, &count, &algo_perf));

    algo = algo_perf.algo;
    conv_algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = static_cast<cudnnConvolutionFwdAlgo_t>(1);
  }

  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));

  void *workspace_data =
      CudnnHandle::GetInstance().GetWorkSpace(workspace_size);
  if (data_type == CUDNN_DATA_DOUBLE) {
    const double alpha_fp64 = static_cast<double>(alpha);
    const double beta_fp64 = static_cast<double>(beta);
    CUDNN_CALL(cudnnConvolutionForward(handle,
                                       &alpha_fp64,
                                       x_desc,
                                       _x,
                                       w_desc,
                                       _w,
                                       conv_desc,
                                       algo,
                                       workspace_data,
                                       workspace_size,
                                       &beta_fp64,
                                       y_desc,
                                       _y));
  } else {
    CUDNN_CALL(cudnnConvolutionForward(handle,
                                       &alpha,
                                       x_desc,
                                       _x,
                                       w_desc,
                                       _w,
                                       conv_desc,
                                       algo,
                                       workspace_data,
                                       workspace_size,
                                       &beta,
                                       y_desc,
                                       _y));
  }

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_conv2d_backward_data(void *v_args,
                                          int num_args,
                                          int format,
                                          float alpha,
                                          float beta,
                                          int input_n,
                                          int input_c,
                                          int input_h,
                                          int input_w,
                                          int filter_n,
                                          int filter_c,
                                          int filter_h,
                                          int filter_w,
                                          int pad_h,
                                          int pad_w,
                                          int stride_h,
                                          int stride_w,
                                          int dilation_h,
                                          int dilation_w,
                                          int groups,
                                          int output_n,
                                          int output_c,
                                          int output_h,
                                          int output_w,
                                          void *stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 3, but received %d.", num_args));
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_w = args[0].operator cinn_buffer_t *()->memory;
  void *_dy = args[1].operator cinn_buffer_t *()->memory;
  void *_dx = args[2].operator cinn_buffer_t *()->memory;

  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);
  cudnnDataType_t data_type = convert_to_cudnn_dtype(v_args, num_args);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc,
                                        data_type,
                                        tensor_format,
                                        filter_n,
                                        filter_c,
                                        filter_h,
                                        filter_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(
      cudnnSetConvolution2dDescriptor(conv_desc,
                                      pad_h,
                                      pad_w,
                                      stride_h,
                                      stride_w,
                                      dilation_h,
                                      dilation_w,
                                      CUDNN_CROSS_CORRELATION,
                                      get_cudnn_compute_dtype(data_type)));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
                                        tensor_format,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  auto &conv_algo_map = ConvAlgoMap::GetInstance();
  std::string hash_key =
      "conv2d backward data, layout=" +
      debug_cudnn_tensor_format(tensor_format) +
      ", dtype=" + debug_cudnn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(filter_n) + "," +
      std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
      std::to_string(filter_w) + "}, output_nchw={" + std::to_string(output_n) +
      "," + std::to_string(output_c) + "," + std::to_string(output_h) + "," +
      std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  int algo_int = conv_algo_map.GetAlgo(hash_key);
  cudnnConvolutionBwdDataAlgo_t algo;
  if (algo_int >= 0) {
    algo = cudnnConvolutionBwdDataAlgo_t(algo_int);
  } else {
    int count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t algo_perf;
    CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithm(
        handle, w_desc, y_desc, conv_desc, x_desc, 1, &count, &algo_perf));

    algo = algo_perf.algo;
    conv_algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  }

  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle, w_desc, y_desc, conv_desc, x_desc, algo, &workspace_size));

  void *workspace_data =
      CudnnHandle::GetInstance().GetWorkSpace(workspace_size);
  if (data_type == CUDNN_DATA_DOUBLE) {
    const double alpha_fp64 = static_cast<double>(alpha);
    const double beta_fp64 = static_cast<double>(beta);
    CUDNN_CALL(cudnnConvolutionBackwardData(handle,
                                            &alpha_fp64,
                                            w_desc,
                                            _w,
                                            y_desc,
                                            _dy,
                                            conv_desc,
                                            algo,
                                            workspace_data,
                                            workspace_size,
                                            &beta_fp64,
                                            x_desc,
                                            _dx));
  } else {
    CUDNN_CALL(cudnnConvolutionBackwardData(handle,
                                            &alpha,
                                            w_desc,
                                            _w,
                                            y_desc,
                                            _dy,
                                            conv_desc,
                                            algo,
                                            workspace_data,
                                            workspace_size,
                                            &beta,
                                            x_desc,
                                            _dx));
  }

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_conv2d_backward_filter(void *v_args,
                                            int num_args,
                                            int format,
                                            float alpha,
                                            float beta,
                                            int input_n,
                                            int input_c,
                                            int input_h,
                                            int input_w,
                                            int filter_n,
                                            int filter_c,
                                            int filter_h,
                                            int filter_w,
                                            int pad_h,
                                            int pad_w,
                                            int stride_h,
                                            int stride_w,
                                            int dilation_h,
                                            int dilation_w,
                                            int groups,
                                            int output_n,
                                            int output_c,
                                            int output_h,
                                            int output_w,
                                            void *stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 3, but received %d.", num_args));
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_dy = args[1].operator cinn_buffer_t *()->memory;
  void *_dw = args[2].operator cinn_buffer_t *()->memory;

  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);
  cudnnDataType_t data_type = convert_to_cudnn_dtype(v_args, num_args);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc,
                                        data_type,
                                        tensor_format,
                                        filter_n,
                                        filter_c,
                                        filter_h,
                                        filter_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(
      cudnnSetConvolution2dDescriptor(conv_desc,
                                      pad_h,
                                      pad_w,
                                      stride_h,
                                      stride_w,
                                      dilation_h,
                                      dilation_w,
                                      CUDNN_CROSS_CORRELATION,
                                      get_cudnn_compute_dtype(data_type)));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
                                        tensor_format,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  auto &algo_map = ConvAlgoMap::GetInstance();
  std::string hash_key =
      "conv2d backward filter, layout=" +
      debug_cudnn_tensor_format(tensor_format) +
      ", dtype=" + debug_cudnn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(filter_n) + "," +
      std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
      std::to_string(filter_w) + "}, output_nchw={" + std::to_string(output_n) +
      "," + std::to_string(output_c) + "," + std::to_string(output_h) + "," +
      std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  int algo_int = algo_map.GetAlgo(hash_key);
  cudnnConvolutionBwdFilterAlgo_t algo;
  if (algo_int >= 0) {
    algo = cudnnConvolutionBwdFilterAlgo_t(algo_int);
  } else {
    int count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t algo_perf;
    CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(
        handle, x_desc, y_desc, conv_desc, w_desc, 1, &count, &algo_perf));

    algo = algo_perf.algo;
    algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  }

  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, x_desc, y_desc, conv_desc, w_desc, algo, &workspace_size));

  void *workspace_data =
      CudnnHandle::GetInstance().GetWorkSpace(workspace_size);
  if (data_type == CUDNN_DATA_DOUBLE) {
    const double alpha_fp64 = static_cast<double>(alpha);
    const double beta_fp64 = static_cast<double>(beta);
    CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
                                              &alpha_fp64,
                                              x_desc,
                                              _x,
                                              y_desc,
                                              _dy,
                                              conv_desc,
                                              algo,
                                              workspace_data,
                                              workspace_size,
                                              &beta_fp64,
                                              w_desc,
                                              _dw));
  } else {
    CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
                                              &alpha,
                                              x_desc,
                                              _x,
                                              y_desc,
                                              _dy,
                                              conv_desc,
                                              algo,
                                              workspace_data,
                                              workspace_size,
                                              &beta,
                                              w_desc,
                                              _dw));
  }

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_pool2d_forward(void *v_args,
                                    int num_args,
                                    int mode,
                                    int format,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int kernel_h,
                                    int kernel_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    void *stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      2,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 2, but received %d.", num_args));
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_y = args[1].operator cinn_buffer_t *()->memory;

  cudnnPoolingMode_t pool_mode = static_cast<cudnnPoolingMode_t>(mode);
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);
  cudnnDataType_t data_type = convert_to_cudnn_dtype(v_args, num_args);

  if (GetCinnCudnnDeterministic() && pool_mode == CUDNN_POOLING_MAX) {
    pool_mode = CUDNN_POOLING_MAX_DETERMINISTIC;
  }

  std::string hash_key =
      "pool2d forward, layout=" + debug_cudnn_tensor_format(tensor_format) +
      ", pool_type=" + debug_cudnn_pool_mode(pool_mode) +
      ", dtype=" + debug_cudnn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, kernel_hw={" + std::to_string(kernel_h) + "," +
      std::to_string(kernel_w) + "}, pad_hw={" + std::to_string(pad_h) + "," +
      std::to_string(pad_w) + "}, stride_hw={" + std::to_string(stride_h) +
      "," + std::to_string(stride_w) + "}, output_nchw={" +
      std::to_string(output_n) + "," + std::to_string(output_c) + "," +
      std::to_string(output_h) + "," + std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  cudnnPoolingDescriptor_t pool_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_desc,
                                         pool_mode,
                                         CUDNN_NOT_PROPAGATE_NAN,
                                         kernel_h,
                                         kernel_w,
                                         pad_h,
                                         pad_w,
                                         stride_h,
                                         stride_w));

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
                                        tensor_format,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  if (data_type == CUDNN_DATA_DOUBLE) {
    const double alpha_fp64 = static_cast<double>(alpha);
    const double beta_fp64 = static_cast<double>(beta);
    CUDNN_CALL(cudnnPoolingForward(
        handle, pool_desc, &alpha_fp64, x_desc, _x, &beta_fp64, y_desc, _y));
  } else {
    CUDNN_CALL(cudnnPoolingForward(
        handle, pool_desc, &alpha, x_desc, _x, &beta, y_desc, _y));
  }

  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_pool2d_backward(void *v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int kernel_h,
                                     int kernel_w,
                                     int pad_h,
                                     int pad_w,
                                     int stride_h,
                                     int stride_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void *stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      4,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 4, but received %d.", num_args));
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_y = args[1].operator cinn_buffer_t *()->memory;
  void *_dy = args[2].operator cinn_buffer_t *()->memory;
  void *_dx = args[3].operator cinn_buffer_t *()->memory;

  cudnnPoolingMode_t pool_mode = static_cast<cudnnPoolingMode_t>(mode);
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);
  cudnnDataType_t data_type = convert_to_cudnn_dtype(v_args, num_args);

  if (GetCinnCudnnDeterministic() && pool_mode == CUDNN_POOLING_MAX) {
    pool_mode = CUDNN_POOLING_MAX_DETERMINISTIC;
  }

  std::string hash_key =
      "pool2d backward, layout=" + debug_cudnn_tensor_format(tensor_format) +
      ", pool_type=" + debug_cudnn_pool_mode(pool_mode) +
      ", dtype=" + debug_cudnn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, kernel_hw={" + std::to_string(kernel_h) + "," +
      std::to_string(kernel_w) + "}, pad_hw={" + std::to_string(pad_h) + "," +
      std::to_string(pad_w) + "}, stride_hw={" + std::to_string(stride_h) +
      "," + std::to_string(stride_w) + ", output_nchw={" +
      std::to_string(output_n) + "," + std::to_string(output_c) + "," +
      std::to_string(output_h) + "," + std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  cudnnPoolingDescriptor_t pool_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_desc,
                                         pool_mode,
                                         CUDNN_NOT_PROPAGATE_NAN,
                                         kernel_h,
                                         kernel_w,
                                         pad_h,
                                         pad_w,
                                         stride_h,
                                         stride_w));

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
                                        tensor_format,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  if (data_type == CUDNN_DATA_DOUBLE) {
    const double alpha_fp64 = static_cast<double>(alpha);
    const double beta_fp64 = static_cast<double>(beta);
    CUDNN_CALL(cudnnPoolingBackward(handle,
                                    pool_desc,
                                    &alpha_fp64,
                                    y_desc,
                                    _y,
                                    y_desc,
                                    _dy,
                                    x_desc,
                                    _x,
                                    &beta_fp64,
                                    x_desc,
                                    _dx));
  } else {
    CUDNN_CALL(cudnnPoolingBackward(handle,
                                    pool_desc,
                                    &alpha,
                                    y_desc,
                                    _y,
                                    y_desc,
                                    _dy,
                                    x_desc,
                                    _x,
                                    &beta,
                                    x_desc,
                                    _dx));
  }

  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_softmax_forward(void *v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void *stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      2,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 2, but received %d.", num_args));
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_y = args[1].operator cinn_buffer_t *()->memory;

  cudnnSoftmaxMode_t softmax_mode = static_cast<cudnnSoftmaxMode_t>(mode);
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);
  cudnnDataType_t data_type = convert_to_cudnn_dtype(v_args, num_args);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
                                        tensor_format,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  if (data_type == CUDNN_DATA_DOUBLE) {
    const double alpha_fp64 = static_cast<double>(alpha);
    const double beta_fp64 = static_cast<double>(beta);
    CUDNN_CALL(cudnnSoftmaxForward(handle,
                                   CUDNN_SOFTMAX_LOG,
                                   softmax_mode,
                                   &alpha_fp64,
                                   x_desc,
                                   _x,
                                   &beta_fp64,
                                   y_desc,
                                   _y));
  } else {
    CUDNN_CALL(cudnnSoftmaxForward(handle,
                                   CUDNN_SOFTMAX_LOG,
                                   softmax_mode,
                                   &alpha,
                                   x_desc,
                                   _x,
                                   &beta,
                                   y_desc,
                                   _y));
  }

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_softmax_backward(void *v_args,
                                      int num_args,
                                      int mode,
                                      int format,
                                      float alpha,
                                      float beta,
                                      int input_n,
                                      int input_c,
                                      int input_h,
                                      int input_w,
                                      int output_n,
                                      int output_c,
                                      int output_h,
                                      int output_w,
                                      void *stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 3, but received %d.", num_args));
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  void *_y = args[0].operator cinn_buffer_t *()->memory;
  void *_dy = args[1].operator cinn_buffer_t *()->memory;
  void *_dx = args[2].operator cinn_buffer_t *()->memory;

  cudnnSoftmaxMode_t softmax_mode = static_cast<cudnnSoftmaxMode_t>(mode);
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);
  cudnnDataType_t data_type = convert_to_cudnn_dtype(v_args, num_args);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
                                        tensor_format,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  if (data_type == CUDNN_DATA_DOUBLE) {
    const double alpha_fp64 = static_cast<double>(alpha);
    const double beta_fp64 = static_cast<double>(beta);
    CUDNN_CALL(cudnnSoftmaxBackward(handle,
                                    CUDNN_SOFTMAX_LOG,
                                    softmax_mode,
                                    &alpha_fp64,
                                    y_desc,
                                    _y,
                                    y_desc,
                                    _dy,
                                    &beta_fp64,
                                    x_desc,
                                    _dx));
  } else {
    CUDNN_CALL(cudnnSoftmaxBackward(handle,
                                    CUDNN_SOFTMAX_LOG,
                                    softmax_mode,
                                    &alpha,
                                    y_desc,
                                    _y,
                                    y_desc,
                                    _dy,
                                    &beta,
                                    x_desc,
                                    _dx));
  }

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

#endif  // CINN_WITH_CUDNN

/********************to be removed in future***********************/

namespace details {

void Gemm(const cublasHandle_t &cublas,
          bool lhs_trans,
          bool rhs_trans,
          const float alpha,
          const float *lhs_data,
          const std::vector<int> &lhs_shape,
          const float *rhs_data,
          const std::vector<int> &rhs_shape,
          const float *bias_data,
          const float beta,
          float *output_data,
          const std::vector<int> &output_shape,
          cudaStream_t stream) {
  int lhs_row = lhs_shape[0];
  int lhs_col = lhs_shape[1];
  int rhs_row = rhs_shape[0];
  int rhs_col = rhs_shape[1];
  int output_row = output_shape[0];
  int output_col = output_shape[1];

  // copy values of bias_data to the output_data
  if (bias_data != nullptr) {
    cudaMemcpyAsync(output_data,
                    bias_data,
                    output_row * output_col * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream);
  }

  int contracting_size = lhs_trans ? lhs_row : lhs_col;
  PADDLE_ENFORCE_EQ(contracting_size,
                    (rhs_trans ? rhs_col : rhs_row),
                    ::common::errors::PreconditionNotMet(
                        "The contracting dimension value of lhs "
                        "matrix should be equal to the "
                        "one of rhs matrix."));
  auto trans_a = rhs_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto trans_b = lhs_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(cublas,
              trans_a,
              trans_b,
              output_col,
              output_row,
              contracting_size,
              &alpha,
              rhs_data,
              rhs_col,
              lhs_data,
              lhs_col,
              &beta,
              output_data,
              output_col);
}

void GemmStridedBatched(const cublasHandle_t &cublas,
                        bool lhs_trans,
                        bool rhs_trans,
                        const float alpha,
                        const float *lhs_data,
                        const std::vector<int> &lhs_shape,
                        const float *rhs_data,
                        const std::vector<int> &rhs_shape,
                        const float *bias_data,
                        const float beta,
                        float *output_data,
                        const std::vector<int> &output_shape,
                        cudaStream_t stream) {
  int lhs_bs = lhs_shape[0];
  int lhs_row = lhs_shape[1];
  int lhs_col = lhs_shape[2];
  int rhs_bs = rhs_shape[0];
  int rhs_row = rhs_shape[1];
  int rhs_col = rhs_shape[2];
  int output_bs = output_shape[0];
  int output_row = output_shape[1];
  int output_col = output_shape[2];
  PADDLE_ENFORCE_EQ(
      lhs_bs,
      rhs_bs,
      ::common::errors::InvalidArgument("bs of lhs and rhs dismatch."));
  PADDLE_ENFORCE_EQ(
      lhs_bs,
      output_bs,
      ::common::errors::InvalidArgument("bs of lhs and output dismatch."));

  // copy values of bias_data to the output_data
  if (bias_data != nullptr) {
    cudaMemcpyAsync(output_data,
                    bias_data,
                    output_bs * output_row * output_col * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream);
  }

  int contracting_size = lhs_trans ? lhs_row : lhs_col;
  PADDLE_ENFORCE_EQ(contracting_size,
                    (rhs_trans ? rhs_col : rhs_row),
                    ::common::errors::PreconditionNotMet(
                        "The contracting dimension value of lhs "
                        "matrix should be equal to the "
                        "one of rhs matrix."));
  auto trans_a = rhs_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto trans_b = lhs_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  int64_t lhs_stride = lhs_row * lhs_col;
  int64_t rhs_stride = rhs_row * rhs_col;
  int64_t output_stride = output_row * output_col;
  cublasSgemmStridedBatched(cublas,
                            trans_a,
                            trans_b,
                            output_col,
                            output_row,
                            contracting_size,
                            &alpha,
                            rhs_data,
                            rhs_col,
                            rhs_stride,
                            lhs_data,
                            lhs_col,
                            lhs_stride,
                            &beta,
                            output_data,
                            output_col,
                            output_stride,
                            output_bs);
}

}  // namespace details

class CusolverHandle {
 public:
  CusolverHandle(const CusolverHandle &) = delete;
  CusolverHandle &operator=(const CusolverHandle &) = delete;
  ~CusolverHandle() { CUSOLVER_CALL(cusolverDnDestroy(handle_)); }
  static CusolverHandle &GetInstance() {
    static CusolverHandle instance;
    return instance;
  }
  cusolverDnHandle_t &GetHandle() { return handle_; }

 private:
  CusolverHandle() { CUSOLVER_CALL(cusolverDnCreate(&handle_)); }
  cusolverDnHandle_t handle_;
};

void cinn_call_cholesky_nvgpu(void *v_args,
                              int num_args,
                              int batch_size,
                              int m,
                              bool upper,
                              void *stream) {
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cinn_buffer_t *x = args[0].operator cinn_buffer_t *();
  cinn_buffer_t *out = args[1].operator cinn_buffer_t *();
  // In cuSOLVER, dense matrix stores in COL_MAJOR, thus FILL_MODE needs to be
  // flipped. See also:
  // https://docs.nvidia.com/cuda/cusolver/index.html#matrix-dense-format
  cublasFillMode_t uplo =
      upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  size_t numel = x->num_elements();
  uint8_t bits = x->type.bits;
  uint8_t bytes = bits / 8;
  PADDLE_ENFORCE_EQ(
      x->type.code,
      cinn_type_code_t::cinn_type_float,
      ::common::errors::InvalidArgument("x's type code (%d) is inequal to %d.",
                                        x->type.code,
                                        cinn_type_code_t::cinn_type_float));
  PADDLE_ENFORCE_EQ(
      bits == 32 || bits == 64,
      true,
      ::common::errors::InvalidArgument(
          "Unsupported bits = %d float data type for cholesky", bits));

  auto cuda_stream = static_cast<cudaStream_t>(stream);

  // Copy data from x to out
  void *x_ptr = reinterpret_cast<void *>(x->memory);
  void *out_ptr = reinterpret_cast<void *>(out->memory);
  CUDA_CALL(cudaMemcpyAsync(
      out_ptr, x_ptr, numel * bytes, cudaMemcpyDeviceToDevice, cuda_stream));
  // Generate pointer array
  thrust::host_vector<void *> host_out_ptr(batch_size, nullptr);
  for (int i = 0; i < batch_size; ++i) {
    host_out_ptr[i] = reinterpret_cast<char *>(out_ptr) + i * m * m * bytes;
  }
  thrust::device_vector<void *> dev_out_ptr(host_out_ptr.begin(),
                                            host_out_ptr.end());
  // Store the return value of each matrix
  thrust::host_vector<int> host_info(batch_size, 0);
  thrust::device_vector<int> dev_info(host_info.begin(), host_info.end());

  cusolverDnHandle_t handler = CusolverHandle::GetInstance().GetHandle();
  CUSOLVER_CALL(cusolverDnSetStream(handler, cuda_stream));
  if (bits == 32) {
    CUSOLVER_CALL(cusolverDnSpotrfBatched(
        handler,
        uplo,
        m,
        reinterpret_cast<float **>(dev_out_ptr.data().get()),
        m,
        thrust::raw_pointer_cast(dev_info.data()),
        batch_size));
  } else if (bits == 64) {
    CUSOLVER_CALL(cusolverDnDpotrfBatched(
        handler,
        uplo,
        m,
        reinterpret_cast<double **>(dev_out_ptr.data().get()),
        m,
        thrust::raw_pointer_cast(dev_info.data()),
        batch_size));
  }

  // Check result
  thrust::copy(dev_info.begin(), dev_info.end(), host_info.begin());
  for (int i = 0; i < host_info.size(); i++) {
    PADDLE_ENFORCE_EQ(host_info[i],
                      0,
                      ::common::errors::PreconditionNotMet(
                          "Cholesky decomposition fail, please check the %d"
                          "th input matrix.",
                          i + 1));
  }
}

void cinn_call_triangular_solve_nvgpu(void *v_args,
                                      int num_args,
                                      int batch_size,
                                      int m,
                                      int k,
                                      bool left_side,
                                      bool upper,
                                      bool transpose_a,
                                      bool unit_diagonal,
                                      void *stream) {
  cublasHandle_t &handle = CublasHandle::GetInstance().GetCublasHandle();
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(handle, custream));

  int b_rows = left_side ? k : m;
  int b_cols = left_side ? m : k;
  int lda = m;
  int ldb = b_rows;
  cublasSideMode_t side = left_side ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo =
      upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasDiagType_t diag =
      unit_diagonal ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cinn_buffer_t *input1 = args[0].operator cinn_buffer_t *();
  cinn_buffer_t *input2 = args[1].operator cinn_buffer_t *();
  cinn_buffer_t *output = args[2].operator cinn_buffer_t *();

  PADDLE_ENFORCE_EQ(input1->type.code,
                    cinn_type_code_t::cinn_type_float,
                    ::common::errors::InvalidArgument(
                        "input1's type code (%d) is inequal to %d.",
                        input1->type.code,
                        cinn_type_code_t::cinn_type_float));
  PADDLE_ENFORCE_EQ(input2->type.code,
                    cinn_type_code_t::cinn_type_float,
                    ::common::errors::InvalidArgument(
                        "input1's type code (%d) is inequal to %d.",
                        input2->type.code,
                        cinn_type_code_t::cinn_type_float));
  PADDLE_ENFORCE_EQ(input1->type.bits,
                    input2->type.bits,
                    ::common::errors::InvalidArgument(
                        "input1 and ipnput2's type bits is dismatch."));
  uint8_t bits = input1->type.bits;
  uint8_t bytes = bits / 8;
  PADDLE_ENFORCE_EQ(
      bits == 32 || bits == 64,
      true,
      ::common::errors::InvalidArgument(
          "Unsupported bits = %d float data type for triangular solve", bits));

  std::string debug_info =
      "triangular solve op: left_side=" + std::to_string(left_side) +
      ", upper=" + std::to_string(uplo) +
      ", transpose_a=" + std::to_string(transa) +
      ", unit_diagonal=" + std::to_string(unit_diagonal) +
      ", batch_size=" + std::to_string(batch_size) +
      ", m=" + std::to_string(m) + ", k=" + std::to_string(k) +
      ", input1_dtype={code: " + std::to_string(input1->type.code) +
      ", bits: " + std::to_string(input1->type.bits) + "}" +
      ", input2_dtype={code: " + std::to_string(input2->type.code) +
      ", bits: " + std::to_string(input2->type.bits) + "}";
  VLOG(4) << debug_info;

  void *a_ptr = reinterpret_cast<void *>(input1->memory);
  void *b_ptr = reinterpret_cast<void *>(input2->memory);
  void *x_ptr = reinterpret_cast<void *>(output->memory);

  // The API cublasStrsmBatched overwrites the right-hand sides, so the
  // right-hand sides should be copied to the output. The output can then be
  // used directly for the calculation.
  size_t numel = input2->num_elements();
  CUDA_CALL(cudaMemcpyAsync(
      x_ptr, b_ptr, numel * bytes, cudaMemcpyDeviceToDevice, custream));

  std::vector<void *> a_array(batch_size, nullptr);
  std::vector<void *> x_array(batch_size, nullptr);
  for (int i = 0; i < batch_size; ++i) {
    a_array[i] = reinterpret_cast<char *>(a_ptr) + i * m * m * bytes;
    x_array[i] = reinterpret_cast<char *>(x_ptr) + i * m * k * bytes;
  }
  thrust::device_vector<void *> dev_a_array(a_array.begin(), a_array.end());
  thrust::device_vector<void *> dev_x_array(x_array.begin(), x_array.end());

  if (bits == 32) {
    std::vector<float> alpha(batch_size, 1.0f);
    CUBLAS_CALL(
        cublasStrsmBatched(handle,
                           side,
                           uplo,
                           transa,
                           diag,
                           b_rows,
                           b_cols,
                           alpha.data(),
                           reinterpret_cast<float **>(dev_a_array.data().get()),
                           lda,
                           reinterpret_cast<float **>(dev_x_array.data().get()),
                           ldb,
                           batch_size));
  } else if (bits == 64) {
    std::vector<double> alpha(batch_size, 1.0);
    CUBLAS_CALL(cublasDtrsmBatched(
        handle,
        side,
        uplo,
        transa,
        diag,
        b_rows,
        b_cols,
        alpha.data(),
        reinterpret_cast<double **>(dev_a_array.data().get()),
        lda,
        reinterpret_cast<double **>(dev_x_array.data().get()),
        ldb,
        batch_size));
  }
}

void cinn_gpu_cublas_mul(const std::vector<int> &attrs,
                         cinn_buffer_t *input1,
                         cinn_buffer_t *input2,
                         cinn_buffer_t *output,
                         cudaStream_t stream) {
  cublasHandle_t &handle = CublasHandle::GetInstance().GetCublasHandle();
  PADDLE_ENFORCE_EQ(input1->type.code,
                    cinn_type_code_t::cinn_type_float,
                    ::common::errors::InvalidArgument(
                        "Expected type code of input is %d, but received %d.",
                        cinn_type_code_t::cinn_type_float,
                        input1->type.code));
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(handle, custream));
  float *x_data = reinterpret_cast<float *>(input1->memory);
  float *y_data = reinterpret_cast<float *>(input2->memory);
  float *out_data = reinterpret_cast<float *>(output->memory);
  int M = 1;
  PADDLE_ENFORCE_GE(attrs.size(),
                    6,
                    ::common::errors::InvalidArgument(
                        "Expected size of attributions is 6, but received %d.",
                        attrs.size()));
  for (int i = 0; i < attrs[attrs.size() - 2]; i++) {
    M *= attrs[i];
  }
  int N = attrs[attrs.size() - 3];
  int K = attrs[attrs.size() - 4];
  float alpha = 1.f;
  float beta = 0.f;
  // M,N * N,K
  cublasSgemm(handle,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              K,
              M,
              N,
              &alpha,
              y_data,
              K,
              x_data,
              N,
              &beta,
              out_data,
              K);
}

void cinn_gpu_cublas_gemm(const std::vector<int> &attrs,
                          cinn_buffer_t *lhs,
                          cinn_buffer_t *rhs,
                          cinn_buffer_t *bias,
                          cinn_buffer_t *output,
                          cudaStream_t stream) {
  cublasHandle_t &handle = CublasHandle::GetInstance().GetCublasHandle();
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(handle, custream));

  PADDLE_ENFORCE_EQ(lhs->type.code,
                    cinn_type_code_t::cinn_type_float,
                    ::common::errors::InvalidArgument(
                        "lhs's type code (%d) is inequal to %d.",
                        lhs->type.code,
                        cinn_type_code_t::cinn_type_float));
  const float *lhs_data = reinterpret_cast<const float *>(lhs->memory);
  const float *rhs_data = reinterpret_cast<const float *>(rhs->memory);
  const float *bias_data =
      bias ? reinterpret_cast<const float *>(bias->memory) : nullptr;
  float *output_data = reinterpret_cast<float *>(output->memory);

  PADDLE_ENFORCE_GE(attrs.size(),
                    13,
                    ::common::errors::InvalidArgument(
                        "Expected size of attributions is greater or "
                        "qeual to 13, but received %d.",
                        attrs.size()));
  int lhs_dim_size = attrs[attrs.size() - 7];
  int rhs_dim_size = attrs[attrs.size() - 6];
  int out_dim_size = attrs[attrs.size() - 5];
  bool lhs_trans = static_cast<bool>(attrs[attrs.size() - 4]);
  bool rhs_trans = static_cast<bool>(attrs[attrs.size() - 3]);
  bool out_trans = static_cast<bool>(attrs[attrs.size() - 2]);
  // 1）C = A^T * B    -->  C^T = B^T * A
  // 2）C = A * B^T    -->  C^T = B * A^T
  // 3）C = A^T * B^T  -->  C^T = B * A
  // 4）C = A * B      -->  C^T = B^T * A^T
  if (out_trans) {
    lhs_trans = static_cast<bool>(attrs[attrs.size() - 3]) ^ out_trans;
    rhs_trans = static_cast<bool>(attrs[attrs.size() - 4]) ^ out_trans;
  }
  const float alpha =
      *reinterpret_cast<const float *>(&attrs[attrs.size() - 1]);
  const float beta = bias ? 1.f : 0.f;
  VLOG(4) << "The lhs_trans value used by cinn_gpu_cublas_gemm: " << lhs_trans;
  VLOG(4) << "The rhs_trans value used by cinn_gpu_cublas_gemm: " << rhs_trans;
  VLOG(4) << "The out_trans value used by cinn_gpu_cublas_gemm: " << out_trans;
  VLOG(4) << "The alpha value used by cinn_gpu_cublas_gemm: " << alpha;
  VLOG(4) << "The beta value used by cinn_gpu_cublas_gemm: " << beta;
  PADDLE_ENFORCE_EQ(lhs_dim_size,
                    rhs_dim_size,
                    ::common::errors::InvalidArgument(
                        "dimension dismatch between lhs and rhs."));
  PADDLE_ENFORCE_EQ(lhs_dim_size,
                    out_dim_size,
                    ::common::errors::InvalidArgument(
                        "dimension dismatch between lhs and out."));
  PADDLE_ENFORCE_EQ(
      (lhs_dim_size == 2 || lhs_dim_size == 3),
      true,
      ::common::errors::InvalidArgument("left operand has 2 or 3 dimension."));

  if (lhs_dim_size == 2) {
    // [row, col]
    std::vector<int> lhs_shape{attrs[0], attrs[1]};
    std::vector<int> rhs_shape{attrs[2], attrs[3]};
    std::vector<int> output_shape{attrs[4], attrs[5]};
    if (out_trans) {
      std::swap(lhs_shape, rhs_shape);
      std::swap(lhs_data, rhs_data);
    }
    details::Gemm(handle,
                  lhs_trans,
                  rhs_trans,
                  alpha,
                  lhs_data,
                  lhs_shape,
                  rhs_data,
                  rhs_shape,
                  bias_data,
                  beta,
                  output_data,
                  output_shape,
                  stream);
  } else {
    // [batch, row, col]
    std::vector<int> lhs_shape{attrs[0], attrs[1], attrs[2]};
    std::vector<int> rhs_shape{attrs[3], attrs[4], attrs[5]};
    std::vector<int> output_shape{attrs[6], attrs[7], attrs[8]};
    if (out_trans) {
      std::swap(lhs_shape, rhs_shape);
      std::swap(lhs_data, rhs_data);
    }
    details::GemmStridedBatched(handle,
                                lhs_trans,
                                rhs_trans,
                                alpha,
                                lhs_data,
                                lhs_shape,
                                rhs_data,
                                rhs_shape,
                                bias_data,
                                beta,
                                output_data,
                                output_shape,
                                stream);
  }
}

class CurandGenerator {
 public:
  CurandGenerator() {
    CURAND_CALL(curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT));
  }

  explicit CurandGenerator(curandRngType rng_type) {
    CURAND_CALL(curandCreateGenerator(&generator_, rng_type));
  }

  ~CurandGenerator() { CURAND_CALL(curandDestroyGenerator(generator_)); }

  curandGenerator_t &GetGenerator() { return generator_; }

  CurandGenerator &SetOffset(uint64_t offset = 0ULL) {
    CURAND_CALL(curandSetGeneratorOffset(generator_, offset));
    VLOG(4) << "Set curand generator offset to: " << offset;
    return *this;
  }

  CurandGenerator &SetSeed(uint64_t seed = 0ULL) {
    // set global seed if seed is zero
    auto rand_seed = (seed == 0ULL) ? RandomSeed::GetOrSet() : seed;
    if (rand_seed != 0ULL && rand_seed != seed_) {
      CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator_, rand_seed));
      VLOG(4) << "Change curand random seed from: " << seed_
              << " to: " << rand_seed;
      seed_ = rand_seed;
    }
    return *this;
  }

  CurandGenerator &SetStream(cudaStream_t stream) {
    if (stream != nullptr && stream != stream_) {
      CURAND_CALL(curandSetStream(generator_, stream));
      VLOG(4) << "Change curand generator stream from: " << stream_
              << " to: " << stream;
      stream_ = stream;
    }
    return *this;
  }

 private:
  curandGenerator_t generator_;
  uint64_t seed_ = 0ULL;
  cudaStream_t stream_ = nullptr;
};

class CurandGeneratorFactory {
 public:
  enum class CurandGeneratorType {
    GENERATOR_DEFAULT,
    GENERATOR_GAUSSIAN,
    GENERATOR_UNIFORM,
    GENERATOR_RANDINT,
  };

  static CurandGenerator &Get(CurandGeneratorType type) {
    switch (type) {
      case CurandGeneratorType::GENERATOR_GAUSSIAN:
        static CurandGenerator gaussian_generator(
            CURAND_RNG_PSEUDO_PHILOX4_32_10);
        return gaussian_generator;
      case CurandGeneratorType::GENERATOR_UNIFORM:
        static CurandGenerator uniform_generator(
            CURAND_RNG_PSEUDO_PHILOX4_32_10);
        return uniform_generator;
      case CurandGeneratorType::GENERATOR_RANDINT:
        static CurandGenerator randint_generator(CURAND_RNG_PSEUDO_MT19937);
        return randint_generator;
      default:
        static CurandGenerator default_generator;
        return default_generator;
    }
  }
};

void cinn_call_gaussian_random(
    void *v_args, int num_args, float mean, float std, int seed, void *stream) {
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cinn_buffer_t *output = args[0].operator cinn_buffer_t *();
  cinn_type_t dtype = output->type;
  size_t numel = output->num_elements();

  curandGenerator_t generator =
      CurandGeneratorFactory::Get(
          CurandGeneratorFactory::CurandGeneratorType::GENERATOR_GAUSSIAN)
          .SetStream(static_cast<cudaStream_t>(stream))
          .SetSeed(seed)
          .GetGenerator();

  VLOG(4) << "cinn_call_gaussian_random: output_size=" << numel
          << ", mean=" << mean << ", std=" << std << ", seed=" << seed;

  if (dtype == cinn_float32_t()) {
    float *ptr = reinterpret_cast<float *>(output->memory);
    CURAND_CALL(curandGenerateNormal(generator, ptr, numel, mean, std));
  } else if (dtype == cinn_float64_t()) {
    double *ptr = reinterpret_cast<double *>(output->memory);
    CURAND_CALL(curandGenerateNormalDouble(generator, ptr, numel, mean, std));
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "gaussian_random only support float32 and float64! Please check."));
  }
}

void cinn_call_uniform_random(
    void *v_args, int num_args, float min, float max, int seed, void *stream) {
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cinn_buffer_t *output = args[0].operator cinn_buffer_t *();
  cinn_type_t dtype = output->type;
  size_t numel = output->num_elements();

  curandGenerator_t generator =
      CurandGeneratorFactory::Get(
          CurandGeneratorFactory::CurandGeneratorType::GENERATOR_UNIFORM)
          .SetStream(static_cast<cudaStream_t>(stream))
          .SetSeed(seed)
          .GetGenerator();

  VLOG(4) << "cinn_call_uniform_random: output_size=" << numel
          << ", min=" << min << ", max=" << max << ", seed=" << seed;

  if (dtype == cinn_float32_t()) {
    float *ptr = reinterpret_cast<float *>(output->memory);
    CURAND_CALL(curandGenerateUniform(generator, ptr, numel));
  } else if (dtype == cinn_float64_t()) {
    double *ptr = reinterpret_cast<double *>(output->memory);
    CURAND_CALL(curandGenerateUniformDouble(generator, ptr, numel));
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "uniform_random only support float32 and float64! Please check."));
  }
}

void cinn_call_randint(void *v_args, int num_args, int seed, void *stream) {
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cinn_buffer_t *output = args[0].operator cinn_buffer_t *();
  cinn_type_t dtype = output->type;
  size_t numel = output->num_elements();

  VLOG(4) << "cinn_call_randint: output_size=" << numel << ", seed=" << seed;

  curandGenerator_t generator =
      CurandGeneratorFactory::Get(
          CurandGeneratorFactory::CurandGeneratorType::GENERATOR_RANDINT)
          .SetStream(static_cast<cudaStream_t>(stream))
          .SetSeed(seed)
          .GetGenerator();

  if (dtype == cinn_int32_t()) {
    unsigned int *ptr = reinterpret_cast<unsigned int *>(output->memory);
    CURAND_CALL(curandGenerate(generator, ptr, numel));
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "randint only support int32! Please check."));
  }
}

#ifdef CINN_WITH_CUDNN

namespace {
cudnnDataType_t convert_to_cudnn_dtype(cinn_buffer_t *input) {
  PADDLE_ENFORCE_NOT_NULL(
      input, ::common::errors::NotFound("the pointer of input is null"));
  auto type_code = input->type.code;
  int bits = input->type.bits;
  cudnnDataType_t data_type;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  if (is_float && bits == 16) {
    data_type = CUDNN_DATA_HALF;
  } else if (is_float && bits == 32) {
    data_type = CUDNN_DATA_FLOAT;
  } else if (is_bfloat16) {
    data_type = CUDNN_DATA_BFLOAT16;
  } else if (is_float && bits == 64) {
    data_type = CUDNN_DATA_DOUBLE;
  } else {
    std::stringstream ss;
    ss << "unsupported cudnn data type: " << static_cast<int>(type_code)
       << ", bits = " << bits;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  return data_type;
}
}  // namespace

#define GetAttrValue(attr_map, key_name, default_value)        \
  int key_name = 0;                                            \
  if (attr_map.count(#key_name) != 0) {                        \
    key_name = attr_map.find(#key_name)->second;               \
  } else if (default_value >= 0) {                             \
    key_name = default_value;                                  \
  } else {                                                     \
    std::stringstream ss;                                      \
    ss << #key_name << " is not exist in attr_map!";           \
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str())); \
  }

void cinn_gpu_cudnn_conv2d(const absl::flat_hash_map<std::string, int> &attr,
                           cinn_buffer_t *x,
                           cinn_buffer_t *w,
                           cinn_buffer_t *y,
                           cudaStream_t stream,
                           cinn::common::Layout target) {
  cudnnTensorFormat_t cudnn_tensor_format;
  if (target == cinn::common::Layout::kNCHW) {
    cudnn_tensor_format = CUDNN_TENSOR_NCHW;
  } else if (target == cinn::common::Layout::kNHWC) {
    cudnn_tensor_format = CUDNN_TENSOR_NHWC;
  } else {
    CINN_NOT_IMPLEMENTED
  }

  GetAttrValue(attr, input_n, -1);
  GetAttrValue(attr, input_c, -1);
  GetAttrValue(attr, input_h, -1);
  GetAttrValue(attr, input_w, -1);
  GetAttrValue(attr, weights_n, -1);
  GetAttrValue(attr, weights_c, -1);
  GetAttrValue(attr, weights_h, -1);
  GetAttrValue(attr, weights_w, -1);
  GetAttrValue(attr, pad_h, 0);
  GetAttrValue(attr, pad_w, 0);
  GetAttrValue(attr, stride_h, 1);
  GetAttrValue(attr, stride_w, 1);
  GetAttrValue(attr, dilation_h, 1);
  GetAttrValue(attr, dilation_w, 1);
  GetAttrValue(attr, groups, 1);
  GetAttrValue(attr, output_n, -1);
  GetAttrValue(attr, output_c, -1);
  GetAttrValue(attr, output_h, -1);
  GetAttrValue(attr, output_w, -1);

  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  void *_x = x->memory;
  void *_w = w->memory;
  void *_y = y->memory;

  auto data_type = convert_to_cudnn_dtype(x);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc,
                                        cudnn_tensor_format,
                                        data_type,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc,
                                        data_type,
                                        cudnn_tensor_format,
                                        weights_n,
                                        weights_c,
                                        weights_h,
                                        weights_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(
      cudnnSetConvolution2dDescriptor(conv_desc,
                                      pad_h,
                                      pad_w,
                                      stride_h,
                                      stride_w,
                                      dilation_h,
                                      dilation_w,
                                      CUDNN_CROSS_CORRELATION,
                                      get_cudnn_compute_dtype(data_type)));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
                                        cudnn_tensor_format,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  auto &conv_algo_map = ConvAlgoMap::GetInstance();
  std::string hash_key =
      "conv2d forward, layout=" + debug_cudnn_tensor_format(CUDNN_TENSOR_NCHW) +
      ", dtype=" + debug_cudnn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(weights_n) + "," +
      std::to_string(weights_c) + "," + std::to_string(weights_h) + "," +
      std::to_string(weights_w) + "}, output_nchw={" +
      std::to_string(output_n) + "," + std::to_string(output_c) + "," +
      std::to_string(output_h) + "," + std::to_string(output_w) + "}";

  cudnnConvolutionFwdAlgo_t algo;
  int algo_int = conv_algo_map.GetAlgo(hash_key);
  if (algo_int >= 0) {
    algo = cudnnConvolutionFwdAlgo_t(algo_int);
  } else {
    int count = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
        handle, x_desc, w_desc, conv_desc, y_desc, 1, &count, &algo_perf));

    algo = algo_perf.algo;
    conv_algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = static_cast<cudnnConvolutionFwdAlgo_t>(1);
  }

  size_t ws_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      handle, x_desc, w_desc, conv_desc, y_desc, algo, &ws_size));

  void *ws_data = CudnnHandle::GetInstance().GetWorkSpace(ws_size);
  if (data_type == CUDNN_DATA_DOUBLE) {
    double alpha[] = {1.f}, beta[] = {0.f};
    CUDNN_CALL(cudnnConvolutionForward(handle,
                                       alpha,
                                       x_desc,
                                       _x,
                                       w_desc,
                                       _w,
                                       conv_desc,
                                       algo,
                                       ws_data,
                                       ws_size,
                                       beta,
                                       y_desc,
                                       _y));
  } else {
    float alpha[] = {1.f}, beta[] = {0.f};
    CUDNN_CALL(cudnnConvolutionForward(handle,
                                       alpha,
                                       x_desc,
                                       _x,
                                       w_desc,
                                       _w,
                                       conv_desc,
                                       algo,
                                       ws_data,
                                       ws_size,
                                       beta,
                                       y_desc,
                                       _y));
  }

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_gpu_cudnn_conv2d_backward_data(
    const absl::flat_hash_map<std::string, int> &attr,
    cinn_buffer_t *w,
    cinn_buffer_t *dy,
    cinn_buffer_t *dx,
    cudaStream_t stream) {
  GetAttrValue(attr, input_n, -1);
  GetAttrValue(attr, input_c, -1);
  GetAttrValue(attr, input_h, -1);
  GetAttrValue(attr, input_w, -1);
  GetAttrValue(attr, weights_n, -1);
  GetAttrValue(attr, weights_c, -1);
  GetAttrValue(attr, weights_h, -1);
  GetAttrValue(attr, weights_w, -1);
  GetAttrValue(attr, pad_h, 0);
  GetAttrValue(attr, pad_w, 0);
  GetAttrValue(attr, stride_h, 1);
  GetAttrValue(attr, stride_w, 1);
  GetAttrValue(attr, dilation_h, 1);
  GetAttrValue(attr, dilation_w, 1);
  GetAttrValue(attr, groups, 1);
  GetAttrValue(attr, output_n, -1);
  GetAttrValue(attr, output_c, -1);
  GetAttrValue(attr, output_h, -1);
  GetAttrValue(attr, output_w, -1);

  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  void *_w = w->memory;
  void *_dy = dy->memory;
  void *_dx = dx->memory;

  auto data_type = convert_to_cudnn_dtype(w);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc,
                                        CUDNN_TENSOR_NCHW,
                                        data_type,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc,
                                        data_type,
                                        CUDNN_TENSOR_NCHW,
                                        weights_n,
                                        weights_c,
                                        weights_h,
                                        weights_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(
      cudnnSetConvolution2dDescriptor(conv_desc,
                                      pad_h,
                                      pad_w,
                                      stride_h,
                                      stride_w,
                                      dilation_h,
                                      dilation_w,
                                      CUDNN_CROSS_CORRELATION,
                                      get_cudnn_compute_dtype(data_type)));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
                                        CUDNN_TENSOR_NCHW,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  auto &conv_algo_map = ConvAlgoMap::GetInstance();
  std::string hash_key =
      "conv2d backward data, layout=" +
      debug_cudnn_tensor_format(CUDNN_TENSOR_NCHW) +
      ", dtype=" + debug_cudnn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(weights_n) + "," +
      std::to_string(weights_c) + "," + std::to_string(weights_h) + "," +
      std::to_string(weights_w) + "}, output_nchw={" +
      std::to_string(output_n) + "," + std::to_string(output_c) + "," +
      std::to_string(output_h) + "," + std::to_string(output_w) + "}";

  int algo_int = conv_algo_map.GetAlgo(hash_key);
  cudnnConvolutionBwdDataAlgo_t algo;
  if (algo_int >= 0) {
    algo = cudnnConvolutionBwdDataAlgo_t(algo_int);
  } else {
    int count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t algo_perf;

    CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithm(
        handle, w_desc, y_desc, conv_desc, x_desc, 1, &count, &algo_perf));

    algo = algo_perf.algo;
    conv_algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  }

  size_t ws_size = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle, w_desc, y_desc, conv_desc, x_desc, algo, &ws_size));

  void *ws_data = CudnnHandle::GetInstance().GetWorkSpace(ws_size);
  if (data_type == CUDNN_DATA_DOUBLE) {
    double alpha[] = {1.0f}, beta[] = {0.0f};
    CUDNN_CALL(cudnnConvolutionBackwardData(handle,
                                            alpha,
                                            w_desc,
                                            _w,
                                            y_desc,
                                            _dy,
                                            conv_desc,
                                            algo,
                                            ws_data,
                                            ws_size,
                                            beta,
                                            x_desc,
                                            _dx));
  } else {
    float alpha[] = {1.0f}, beta[] = {0.0f};
    CUDNN_CALL(cudnnConvolutionBackwardData(handle,
                                            alpha,
                                            w_desc,
                                            _w,
                                            y_desc,
                                            _dy,
                                            conv_desc,
                                            algo,
                                            ws_data,
                                            ws_size,
                                            beta,
                                            x_desc,
                                            _dx));
  }

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_gpu_cudnn_conv2d_backward_filter(
    const absl::flat_hash_map<std::string, int> &attr,
    cinn_buffer_t *x,
    cinn_buffer_t *dy,
    cinn_buffer_t *dw,
    cudaStream_t stream) {
  GetAttrValue(attr, input_n, -1);
  GetAttrValue(attr, input_c, -1);
  GetAttrValue(attr, input_h, -1);
  GetAttrValue(attr, input_w, -1);
  GetAttrValue(attr, weights_n, -1);
  GetAttrValue(attr, weights_c, -1);
  GetAttrValue(attr, weights_h, -1);
  GetAttrValue(attr, weights_w, -1);
  GetAttrValue(attr, pad_h, 0);
  GetAttrValue(attr, pad_w, 0);
  GetAttrValue(attr, stride_h, 1);
  GetAttrValue(attr, stride_w, 1);
  GetAttrValue(attr, dilation_h, 1);
  GetAttrValue(attr, dilation_w, 1);
  GetAttrValue(attr, groups, 1);
  GetAttrValue(attr, output_n, -1);
  GetAttrValue(attr, output_c, -1);
  GetAttrValue(attr, output_h, -1);
  GetAttrValue(attr, output_w, -1);

  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));

  void *_x = x->memory;
  void *_dy = dy->memory;
  void *_dw = dw->memory;

  auto data_type = convert_to_cudnn_dtype(x);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc,
                                        CUDNN_TENSOR_NCHW,
                                        data_type,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc,
                                        data_type,
                                        CUDNN_TENSOR_NCHW,
                                        weights_n,
                                        weights_c,
                                        weights_h,
                                        weights_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(
      cudnnSetConvolution2dDescriptor(conv_desc,
                                      pad_h,
                                      pad_w,
                                      stride_h,
                                      stride_w,
                                      dilation_h,
                                      dilation_w,
                                      CUDNN_CROSS_CORRELATION,
                                      get_cudnn_compute_dtype(data_type)));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
                                        CUDNN_TENSOR_NCHW,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  auto &algo_map = ConvAlgoMap::GetInstance();
  std::string hash_key =
      "conv2d backward filter, layout=" +
      debug_cudnn_tensor_format(CUDNN_TENSOR_NCHW) +
      ", dtype=" + debug_cudnn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(weights_n) + "," +
      std::to_string(weights_c) + "," + std::to_string(weights_h) + "," +
      std::to_string(weights_w) + "}, output_nchw={" +
      std::to_string(output_n) + "," + std::to_string(output_c) + "," +
      std::to_string(output_h) + "," + std::to_string(output_w) + "}";

  int algo_int = algo_map.GetAlgo(hash_key);
  cudnnConvolutionBwdFilterAlgo_t algo;
  if (algo_int >= 0) {
    algo = cudnnConvolutionBwdFilterAlgo_t(algo_int);
  } else {
    int count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t algo_perf;
    CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(
        handle, x_desc, y_desc, conv_desc, w_desc, 1, &count, &algo_perf));

    algo = algo_perf.algo;
    algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  }

  size_t ws_size = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, x_desc, y_desc, conv_desc, w_desc, algo, &ws_size));

  void *ws_data = CudnnHandle::GetInstance().GetWorkSpace(ws_size);
  if (data_type == CUDNN_DATA_DOUBLE) {
    double alpha[] = {1.0}, beta[] = {0.0};
    CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
                                              alpha,
                                              x_desc,
                                              _x,
                                              y_desc,
                                              _dy,
                                              conv_desc,
                                              algo,
                                              ws_data,
                                              ws_size,
                                              beta,
                                              w_desc,
                                              _dw));
  } else {
    float alpha[] = {1.0}, beta[] = {0.0};
    CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
                                              alpha,
                                              x_desc,
                                              _x,
                                              y_desc,
                                              _dy,
                                              conv_desc,
                                              algo,
                                              ws_data,
                                              ws_size,
                                              beta,
                                              w_desc,
                                              _dw));
  }

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_gpu_cudnn_pool2d(const std::vector<int> &attrs,
                           const std::vector<std::string> &str_attrs,
                           cinn_buffer_t *input,
                           cinn_buffer_t *output,
                           cudaStream_t stream) {
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  PADDLE_ENFORCE_EQ(attrs.size(),
                    17,
                    ::common::errors::InvalidArgument(
                        "Expected size of attributions is 17, but received %d.",
                        attrs.size()));
  // Here the input paddings are pad_top, pad_bottom, pad_left, pad_right.
  // Since pad_top==pad_bottom and pad_left==pad_rifht, we only take pad_top and
  // pad_left.
  int input_n = attrs[0];
  int input_c = attrs[1];
  int input_h = attrs[2];
  int input_w = attrs[3];
  int kernel_h = attrs[4];
  int kernel_w = attrs[5];
  int pad_h = attrs[6];
  int pad_w = attrs[8];
  int stride_h = attrs[10];
  int stride_w = attrs[11];
  int output_n = attrs[12];
  int output_c = attrs[13];
  int output_h = attrs[14];
  int output_w = attrs[15];
  int adaptive = attrs[16];
  std::string pool_type = str_attrs[0];
  cudnnPoolingDescriptor_t pooling_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc));
  cudnnPoolingMode_t pool_mode;
  if (pool_type == "max") {
    pool_mode = CUDNN_POOLING_MAX;
  } else if (pool_type == "avg") {
    pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
  }
  if (adaptive == 1) {
    stride_h = input_h / output_h;
    stride_w = input_w / output_w;
    kernel_h = input_h - (output_h - 1) * stride_h;
    kernel_w = input_w - (output_w - 1) * stride_w;
  }

  auto data_type = convert_to_cudnn_dtype(input);

  CUDNN_CALL(cudnnSetPooling2dDescriptor(pooling_desc,
                                         pool_mode,
                                         CUDNN_NOT_PROPAGATE_NAN,
                                         kernel_h,
                                         kernel_w,
                                         pad_h,
                                         pad_w,
                                         stride_h,
                                         stride_w));

  cudnnTensorDescriptor_t in_desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc,
                                        CUDNN_TENSOR_NCHW,
                                        data_type,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));

  cudnnTensorDescriptor_t out_desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc,
                                        CUDNN_TENSOR_NCHW,
                                        data_type,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));

  void *in_data = input->memory;
  void *out_data = output->memory;

  if (data_type == CUDNN_DATA_DOUBLE) {
    double alpha = 1.0f;
    double beta = 0.0f;
    CUDNN_CALL(cudnnPoolingForward(handle,
                                   pooling_desc,
                                   &alpha,
                                   in_desc,
                                   in_data,
                                   &beta,
                                   out_desc,
                                   out_data));
  } else {
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnPoolingForward(handle,
                                   pooling_desc,
                                   &alpha,
                                   in_desc,
                                   in_data,
                                   &beta,
                                   out_desc,
                                   out_data));
  }

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
}

void infer_shape_set_value(int row, int col, int64_t value, int64_t **v) {
  v[row][col] = value;
}
void cinn_gpu_cudnn_softmax(const std::vector<int> &attrs,
                            cinn_buffer_t *input,
                            cinn_buffer_t *output,
                            cudaStream_t stream) {
  std::vector<int> shape;
  int rank = attrs.size() - 1;
  for (int i = 0; i < rank; i++) {
    shape.push_back(attrs[i]);
  }
  int axis = attrs.back();
  axis = axis < 0 ? rank + axis : axis;
  int inner_num = 1;
  int outer_num = 1;
  for (int i = 0; i < shape.size(); i++) {
    if (i < axis)
      outer_num *= shape[i];
    else if (i > axis)
      inner_num *= shape[i];
  }
  rank = shape.size();

  auto data_type = convert_to_cudnn_dtype(input);

  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  void *in_data = input->memory;
  void *out_data = output->memory;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc,
                                        CUDNN_TENSOR_NCHW,
                                        data_type,
                                        outer_num,
                                        shape[axis],
                                        inner_num,
                                        1));

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc,
                                        CUDNN_TENSOR_NCHW,
                                        data_type,
                                        outer_num,
                                        shape[axis],
                                        inner_num,
                                        1));

  if (data_type == CUDNN_DATA_DOUBLE) {
    double alpha = 1.f;
    double beta = 0.f;
    CUDNN_CALL(cudnnSoftmaxForward(handle,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha,
                                   in_desc,
                                   in_data,
                                   &beta,
                                   out_desc,
                                   out_data));
  } else {
    float alpha = 1.f;
    float beta = 0.f;
    CUDNN_CALL(cudnnSoftmaxForward(handle,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha,
                                   in_desc,
                                   in_data,
                                   &beta,
                                   out_desc,
                                   out_data));
  }

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
}

#endif  // CINN_WITH_CUDNN

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
