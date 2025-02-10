/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/common/float8_e5m2.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/kernels/funcs/blas/blaslt_gemm_search.h"

namespace dyl = phi::dynload;

namespace phi {
namespace fusion {
namespace cutlass_internal {

#define PADDLE_CUBLASLT_STATUS_CHECK(name)                                    \
  PADDLE_ENFORCE_EQ(                                                          \
      status,                                                                 \
      CUBLAS_STATUS_SUCCESS,                                                  \
      common::errors::External(                                               \
          #name                                                               \
          "execution error"                                                   \
          "refer https://docs.nvidia.com/cuda/cublas/index.html to get more " \
          "information"))

template <typename T>
inline cudaDataType_t GetCublasLtDataType() {
  return CUDA_R_32F;
}

template <>
inline cudaDataType_t GetCublasLtDataType<phi::dtype::float16>() {
  return CUDA_R_16F;
}

template <>
inline cudaDataType_t GetCublasLtDataType<phi::dtype::bfloat16>() {
  return CUDA_R_16BF;
}

template <typename T>
void CublasLtMatmulFP8(const phi::GPUContext& dev_ctx,
                       const int batch_count,
                       const int m,
                       const int n,
                       const int k,
                       const phi::DenseTensor& mat_a,
                       const phi::DenseTensor& mat_b,
                       const float scale,
                       const paddle::optional<DenseTensor>& bias,
                       const std::string& activation_type,
                       phi::DenseTensor* out) {
  // init data structure
  cublasStatus_t status;
  auto A_type = CUDA_R_8F_E4M3;
  auto B_type = CUDA_R_8F_E4M3;
  auto Bias_type = GetCublasLtDataType<T>();
  auto C_type = GetCublasLtDataType<T>();

  cublasLtMatmulDesc_t matmul_desc_;
  cublasLtMatrixLayout_t A_desc_;
  cublasLtMatrixLayout_t B_desc_;
  cublasLtMatrixLayout_t Bias_desc_;
  cublasLtMatrixLayout_t C_desc_;
  float alpha_ = scale;
  float beta_ = 0.0f;

  cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32F;
  status =
      dyl::cublasLtMatmulDescCreate(&matmul_desc_, cudaComputeType, CUDA_R_32F);
  PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescCreate);

  cublasOperation_t op_transpose = CUBLAS_OP_T;
  status = dyl::cublasLtMatmulDescSetAttribute(matmul_desc_,
                                               CUBLASLT_MATMUL_DESC_TRANSA,
                                               &op_transpose,
                                               sizeof(op_transpose));
  PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);

  // int8_t fast_accum = 1;
  // status = dyl::cublasLtMatmulDescSetAttribute(matmul_desc_,
  //                                              CUBLASLT_MATMUL_DESC_FAST_ACCUM,
  //                                              &fast_accum,
  //                                              sizeof(fast_accum));
  // PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);

  cublasLtEpilogue_t epilogue;
  const T* bias_ptr = nullptr;
  if (bias) {
    beta_ = 1.0f;
    bias_ptr = const_cast<T*>(bias.get().data<T>());
  }
  if (activation_type == "gelu") {
    epilogue = CUBLASLT_EPILOGUE_GELU;
    status = dyl::cublasLtMatmulDescSetAttribute(matmul_desc_,
                                                 CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                 &epilogue,
                                                 sizeof(epilogue));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);
  } else if (activation_type == "relu") {
    epilogue = CUBLASLT_EPILOGUE_RELU;
    status = dyl::cublasLtMatmulDescSetAttribute(matmul_desc_,
                                                 CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                 &epilogue,
                                                 sizeof(epilogue));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);
  } else if (activation_type == "identity") {
    VLOG(3) << "No activation function set, the activation type is identity";
  } else {
    PADDLE_THROW(common::errors::Fatal(
        "Can not support this activation type, please check the act"));
  }

  status = dyl::cublasLtMatrixLayoutCreate(&B_desc_, B_type, k, n, k);
  status = dyl::cublasLtMatrixLayoutCreate(&A_desc_, A_type, k, m, k);
  status = dyl::cublasLtMatrixLayoutCreate(&Bias_desc_, Bias_type, n, m, 0);
  status = dyl::cublasLtMatrixLayoutCreate(&C_desc_, C_type, n, m, n);
  PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatrixLayoutCreate);

  if (batch_count > 1) {
    int64_t strideb = n * k;
    int64_t stridea = m * k;
    int64_t stridebias = 0;
    int64_t stridec = m * n;
    status = dyl::cublasLtMatrixLayoutSetAttribute(
        B_desc_,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);
    status = dyl::cublasLtMatrixLayoutSetAttribute(
        B_desc_,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &strideb,
        sizeof(strideb));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);
    status = dyl::cublasLtMatrixLayoutSetAttribute(
        A_desc_,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);
    status = dyl::cublasLtMatrixLayoutSetAttribute(
        A_desc_,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stridea,
        sizeof(stridea));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);
    status = dyl::cublasLtMatrixLayoutSetAttribute(
        Bias_desc_,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);
    status = dyl::cublasLtMatrixLayoutSetAttribute(
        Bias_desc_,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stridebias,
        sizeof(stridebias));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);
    status = dyl::cublasLtMatrixLayoutSetAttribute(
        C_desc_,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);
    status = dyl::cublasLtMatrixLayoutSetAttribute(
        C_desc_,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stridec,
        sizeof(stridec));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);
  }

  cublasLtMatmulAlgo_t* algo =
      funcs::cublaslt_internal::CublasLtAlgoCache::Instance()
          .CublasLtAlgoSelect(dev_ctx.cublaslt_handle(),
                              m,
                              n,
                              k,
                              batch_count,
                              mat_b.data<phi::dtype::float8_e4m3fn>(),
                              mat_a.data<phi::dtype::float8_e4m3fn>(),
                              bias_ptr,
                              out->data<T>(),
                              &alpha_,
                              &beta_,
                              matmul_desc_,
                              B_desc_,
                              A_desc_,
                              Bias_desc_,
                              C_desc_,
                              CUBLAS_COMPUTE_32F,
                              CUDA_R_32F,
                              B_type,
                              A_type,
                              Bias_type,
                              C_type,
                              dev_ctx.stream());

  if (algo == nullptr) {
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtMatmulPreference_t preference = NULL;

    size_t workspace_size = 64 * 1024 * 1024;
    status = dyl::cublasLtMatmulPreferenceCreate(&preference);
    status = dyl::cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size));
    status = dyl::cublasLtMatmulAlgoGetHeuristic(dev_ctx.cublaslt_handle(),
                                                 matmul_desc_,
                                                 B_desc_,
                                                 A_desc_,
                                                 Bias_desc_,
                                                 C_desc_,
                                                 preference,
                                                 1,
                                                 &heuristicResult,
                                                 &returnedResults);

    PADDLE_ENFORCE_NE(returnedResults,
                      0,
                      common::errors::NotFound(
                          "Unable to find suitable cuBLAS GEMM algorithm"));
    algo = &heuristicResult.algo;
  }

  cublasLtMatmulHeuristicResult_t heurResult;
  status = dyl::cublasLtMatmulAlgoCheck(dev_ctx.cublaslt_handle(),
                                        matmul_desc_,
                                        B_desc_,
                                        A_desc_,
                                        Bias_desc_,
                                        C_desc_,
                                        algo,
                                        &heurResult);
  PADDLE_ENFORCE_EQ(
      status,
      CUBLAS_STATUS_SUCCESS,
      common::errors::External("cuBLAS GEMM algorithm check failed"));

  size_t temp_workspace_size = heurResult.workspaceSize;
  auto temp_workspace = phi::memory_utils::Alloc(
      phi::GPUPlace(backends::gpu::GetCurrentDeviceId()), temp_workspace_size);

  status = dyl::cublasLtMatmul(dev_ctx.cublaslt_handle(),
                               matmul_desc_,
                               &alpha_,
                               mat_b.data<phi::dtype::float8_e4m3fn>(),
                               B_desc_,
                               mat_a.data<phi::dtype::float8_e4m3fn>(),
                               A_desc_,
                               &beta_,
                               bias_ptr,
                               Bias_desc_,
                               out->data<T>(),
                               C_desc_,
                               algo,
                               temp_workspace->ptr(),  // NOLINT
                               temp_workspace_size,
                               dev_ctx.stream());
  PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmul);
}

template <typename Context>
void cublaslt_fp8_fp8_fp16_gemm(
    const Context& ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const paddle::optional<DenseTensor>& bias,
    bool transpose_x,
    bool transpose_y,
    const float scale,  // only support per-tensor quantization
    const std::string& activation_type,
    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(x.dims().size() == y.dims().size(),
                    true,
                    common::errors::InvalidArgument(
                        "FP8 gemm x_dims.size, must equal to y_dims.size,"
                        "x_dims.size = %d, but y_dims.size = %d",
                        x.dims().size(),
                        y.dims().size()));

  int rank = x.dims().size();
  int m = transpose_x ? x.dims()[rank - 1] : x.dims()[rank - 2];
  int n = transpose_y ? y.dims()[rank - 2] : y.dims()[rank - 1];
  int k = transpose_x ? x.dims()[rank - 2] : x.dims()[rank - 1];

  int y_k = transpose_y ? y.dims()[rank - 1] : y.dims()[rank - 2];
  PADDLE_ENFORCE_EQ(
      k == y_k,
      true,
      common::errors::InvalidArgument(
          "FP8 gemm x_k needs to equal to y_k, x_k = %d, but y_k = %d",
          k,
          y_k));

  if (bias) {
    PADDLE_ENFORCE_EQ(bias->dims()[0] == n,
                      true,
                      common::errors::InvalidArgument(
                          "FP8 gemm bias_vector_dim needs to equal "
                          "to n, n = %d, but bias_vector_dim = %d",
                          n,
                          bias->dims()[0]));
  }

  PADDLE_ENFORCE_EQ(k % 16 == 0,
                    true,
                    common::errors::InvalidArgument(
                        "FP8 gemm need k % 16 = 0, but k = %d", k));

  ctx.template Alloc<phi::dtype::float16>(out);
  int batch_count = 1;
  for (size_t i = 0; i < rank - 2; ++i) {
    batch_count *= x.dims()[i];
  }
  CublasLtMatmulFP8<phi::dtype::float16>(
      ctx, batch_count, m, n, k, x, y, scale, bias, activation_type, out);
}

template <typename Context>
void cublaslt_fp8_fp8_bf16_gemm(
    const Context& ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const paddle::optional<DenseTensor>& bias,
    bool transpose_x,
    bool transpose_y,
    const float scale,  // only support per-tensor quantization
    const std::string& activation_type,
    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(x.dims().size() == y.dims().size(),
                    true,
                    common::errors::InvalidArgument(
                        "FP8 gemm x_dims.size, must equal to y_dims.size,"
                        "x_dims.size = %d, but y_dims.size = %d",
                        x.dims().size(),
                        y.dims().size()));

  int rank = x.dims().size();
  int m = transpose_x ? x.dims()[rank - 1] : x.dims()[rank - 2];
  int n = transpose_y ? y.dims()[rank - 2] : y.dims()[rank - 1];
  int k = transpose_x ? x.dims()[rank - 2] : x.dims()[rank - 1];

  int y_k = transpose_y ? y.dims()[rank - 1] : y.dims()[rank - 2];
  PADDLE_ENFORCE_EQ(
      k == y_k,
      true,
      common::errors::InvalidArgument(
          "FP8 gemm x_k needs to equal to y_k, x_k = %d, but y_k = %d",
          k,
          y_k));

  if (bias) {
    PADDLE_ENFORCE_EQ(bias->dims()[0] == n,
                      true,
                      common::errors::InvalidArgument(
                          "FP8 gemm bias_vector_dim needs to equal "
                          "to n, n = %d, but bias_vector_dim = %d",
                          n,
                          bias->dims()[0]));
  }

  PADDLE_ENFORCE_EQ(k % 16 == 0,
                    true,
                    common::errors::InvalidArgument(
                        "FP8 gemm need k % 16 = 0, but k = %d", k));

  ctx.template Alloc<phi::dtype::bfloat16>(out);
  int batch_count = 1;
  for (size_t i = 0; i < rank - 2; ++i) {
    batch_count *= x.dims()[i];
  }
  CublasLtMatmulFP8<phi::dtype::bfloat16>(
      ctx, batch_count, m, n, k, x, y, scale, bias, activation_type, out);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
