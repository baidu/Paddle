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
#pragma once
#include "../fp8_common.h"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "dual_gemm/device/dual_gemm.h"
#include "dual_gemm/thread/left_gelu_and_mul.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

template <typename InputType, typename BiasType, typename OutType>
bool dispatch_dual_gemm_scale_bias_geglu(DualGemmEpilogueAllParams params) {
  using ElementInputA = typename std::conditional_t<
      std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;
  using ElementInputB = typename std::conditional_t<
      std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;
  using ElementInputC = typename std::conditional_t<
      std::is_same_v<BiasType, phi::dtype::bfloat16>,
      cutlass::bfloat16_t,
      cutlass::half_t>;
  using ElementOutput = typename std::conditional_t<
      std::is_same_v<OutType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementComputeEpilogue = float;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;
  static int const kAlignmentA = 16;
  static int const kAlignmentB = 16;

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm89;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<64, 64, 64>;  // <- threadblock tile M = 64, N
                                             // = 64, K = 64
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp =
      cutlass::gemm::GemmShape<32, 32, 64>;  // <- warp tile M = 32, N = 32, K =
                                             // 64
  // This code section describes the size of MMA op
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;  // <- MMA Op tile M =
                                                           // 16, N = 8, K = 32

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

  using EpilogueOp0 = cutlass::epilogue::thread::LinearCombination<
      ElementInputC,  // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementInputC>::
                value,     // <- the number of elements per vectorized
                           // memory access. For a byte, it's 16
                           // elements. This becomes the vector width of
                           // math instructions in the epilogue too
      ElementAccumulator,  // <- data type of accumulator
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::
          NoBetaScaling>;  // <- data type for alpha/beta in linear
                           // combination function

  using EpilogueOp1 = cutlass::epilogue::thread::LeftGELUAndMul<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementInputC>::value,
      ElementInputC,
      ElementCompute>;

  // Number of pipelines you want to use
  constexpr int NumStages = 3;
  constexpr bool StoreD0 = false;
  constexpr bool StoreD1 = false;
  constexpr bool SplitKSerial = false;

  using Gemm = cutlass::gemm::device::DualGemm<ElementInputA,
                                               LayoutInputA,
                                               ElementInputB,
                                               LayoutInputB,
                                               LayoutInputB,
                                               ElementInputC,
                                               ElementOutput,
                                               LayoutOutput,
                                               ElementAccumulator,
                                               MMAOp,
                                               SmArch,
                                               ShapeMMAThreadBlock,
                                               ShapeMMAWarp,
                                               ShapeMMAOp,
                                               EpilogueOp0,
                                               EpilogueOp0,
                                               EpilogueOp1,
                                               SwizzleThreadBlock,
                                               NumStages,
                                               StoreD0,
                                               StoreD1,
                                               SplitKSerial,
                                               kAlignmentA,
                                               kAlignmentB,
                                               cutlass::arch::OpMultiplyAdd>;

  cutlass::gemm::GemmCoord problem_size =
      cutlass::gemm::GemmCoord{params.M, params.N, params.K};

  cutlass::gemm::DualGemmMode mode = cutlass::gemm::DualGemmMode::kBatched;

  typename cutlass::TensorRef<typename Gemm::ElementC, typename Gemm::LayoutC>
      nullptr_ref{};
  int split_k_slices = Gemm::kSplitKSerial ? 2 : 1;

  typename Gemm::Arguments arguments{
      mode,
      problem_size,
      {reinterpret_cast<ElementInputA*>(const_cast<void*>(params.A)),
       params.lda},
      {reinterpret_cast<ElementInputB*>(const_cast<void*>(params.B0)),
       params.ldb},
      {reinterpret_cast<ElementInputC*>(const_cast<void*>(params.bias0)), 0},
      nullptr_ref,
      {reinterpret_cast<ElementInputB*>(const_cast<void*>(params.B1)),
       params.ldb},
      {reinterpret_cast<ElementInputC*>(const_cast<void*>(params.bias1)), 0},
      nullptr_ref,
      {reinterpret_cast<ElementOutput*>(const_cast<void*>(params.D)),
       params.ldd},
      {params.scale0},
      {params.scale1},
      {params.scale_out},
      split_k_slices,  // split_k_slices
      params.batch_count,
      params.lda * params.M,
      params.ldb * params.N,
      params.ldb * params.N,
      0,
      params.ldd * params.M,
  };

  Gemm gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::can_implement() failed" << std::endl;
    return false;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  phi::Allocator* allocator = paddle::GetAllocator(params.place);
  auto workspace = allocator->Allocate(workspace_size);

  //
  // Run the GEMM
  //
  status = gemm_op(arguments, workspace->ptr(), params.stream);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::run() failed" << std::endl;
    return false;
  }

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
    return false;
  }
  return true;
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
