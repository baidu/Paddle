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

#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include "paddle/phi/backends/dynload/hipblasLt.h"
#include "paddle/phi/core/dense_tensor.h"

namespace dyl = phi::dynload;

namespace phi {
#if HIP_VERSION >= 60000000
#define HIP_DATATYPE_R_32I HIP_R_32I
#define HIP_DATATYPE_R_8I HIP_R_8I
#define HIP_DATATYPE_R_32F HIP_R_32F
#define HIP_DATATYPE_R_16BF HIP_R_16BF
#define HIP_DATATYPE_R_16F HIP_R_16F
#endif

class CublasLtHelper {
 public:
  CublasLtHelper(int m, int k, int n, hipblasLtHandle_t handle)
      : handle_(handle), alpha_(1), beta_(0), m_(m), k_(k), n_(n) {
    hipblasStatus_t status;
    hipblasComputeType_t hipComputeType = HIPBLAS_COMPUTE_32I;

    // matmul desc
    status = dyl::hipblasLtMatmulDescCreate(
        &matmul_desc_, hipComputeType, HIP_DATATYPE_R_32I);

    PADDLE_ENFORCE_EQ(
        status,
        HIPBLAS_STATUS_SUCCESS,
        common::errors::External("hipblasLtMatmulDescCreate execution error"));
    hipblasOperation_t op_transpose = HIPBLAS_OP_T;
    status = dyl::hipblasLtMatmulDescSetAttribute(matmul_desc_,
                                                  HIPBLASLT_MATMUL_DESC_TRANSA,
                                                  &op_transpose,
                                                  sizeof(op_transpose));
    PADDLE_ENFORCE_EQ(status,
                      HIPBLAS_STATUS_SUCCESS,
                      common::errors::External(
                          "hipblasLtMatmulDescSetAttribute execution error"));

    // matrix desc
    status =
        dyl::hipblasLtMatrixLayoutCreate(&B_desc_, HIP_DATATYPE_R_8I, k, n, k);
    PADDLE_ENFORCE_EQ(status,
                      HIPBLAS_STATUS_SUCCESS,
                      common::errors::External(
                          "hipblasLtMatrixLayoutCreate execution error"));

    status =
        dyl::hipblasLtMatrixLayoutCreate(&A_desc_, HIP_DATATYPE_R_8I, k, m, k);
    PADDLE_ENFORCE_EQ(status,
                      HIPBLAS_STATUS_SUCCESS,
                      common::errors::External(
                          "hipblasLtMatrixLayoutCreate execution error"));

    status =
        dyl::hipblasLtMatrixLayoutCreate(&C_desc_, HIP_DATATYPE_R_32I, n, m, n);
    PADDLE_ENFORCE_EQ(status,
                      HIPBLAS_STATUS_SUCCESS,
                      common::errors::External(
                          "hipblasLtMatrixLayoutCreate execution error"));
  }
  ~CublasLtHelper() {}

  void GEMM(const int8_t* A_dev,
            const int8_t* B_dev,
            int32_t* C_dev,
            hipStream_t stream,
            void* workspace = nullptr) {
    hipblasStatus_t status;

    status = dyl::hipblasLtMatmul(handle_,
                                  matmul_desc_,
                                  &alpha_,
                                  B_dev,
                                  B_desc_,
                                  A_dev,
                                  A_desc_,
                                  &beta_,
                                  C_dev,
                                  C_desc_,
                                  C_dev,
                                  C_desc_,
                                  &algo_,
                                  workspace,
                                  workspace_size_,
                                  stream);
    PADDLE_ENFORCE_EQ(
        status,
        HIPBLAS_STATUS_SUCCESS,
        common::errors::External("cublasLtMatmul execution error"));
  }

 private:
  hipblasLtHandle_t handle_;
  hipblasLtMatmulDesc_t matmul_desc_;
  hipblasLtMatrixLayout_t A_desc_;
  hipblasLtMatrixLayout_t B_desc_;
  hipblasLtMatrixLayout_t C_desc_;

  hipblasLtMatmulAlgo_t algo_;

  int32_t alpha_ = 1;
  int32_t beta_ = 0;

  int m_ = 0;
  int k_ = 0;
  int n_ = 0;

  size_t workspace_size_ = 0;
};

template <typename T>
inline hipDataType_t GetCublasLtDataType() {
  return HIP_DATATYPE_R_32F;
}

template <>
inline hipDataType_t GetCublasLtDataType<phi::dtype::float16>() {
  return HIP_DATATYPE_R_16F;
}

template <>
inline hipDataType_t GetCublasLtDataType<phi::dtype::bfloat16>() {
  return HIP_DATATYPE_R_16BF;
}

template <typename T>
void CublasLtMatmulFP8(const phi::GPUContext& dev_ctx,
                       const phi::DenseTensor& mat_a,
                       const phi::DenseTensor& mat_b,
                       phi::DenseTensor* workspace,
                       phi::DenseTensor* out) {
  PADDLE_THROW(common::errors::Unimplemented(
      "FP8 matmul is not supported on HIP platform."));
}
}  // namespace phi
