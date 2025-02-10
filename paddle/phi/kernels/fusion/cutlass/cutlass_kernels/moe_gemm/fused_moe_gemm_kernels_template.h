/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <optional>
#include <sstream>
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/datatype_traits.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "paddle/common/errors.h"
#include "paddle/phi/core/enforce.h"

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/compute_occupancy.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/epilogue_helpers.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/threadblock/default_mma.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/fused_moe_cutlass_kernel.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/fused_moe_gemm_kernels.h"

#pragma GCC diagnostic pop

#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/cutlass_heuristic.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/gemm_config_manager.h"
#include "paddle/phi/kernels/fusion/cutlass/utils/cuda_utils.h"

namespace phi {

// ============================= Variable batched Gemm things
// ===========================
template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
void generic_moe_gemm_kernelLauncher(const T* A,
                                     const WeightType* B,
                                     const T* weight_scales,
                                     const T* biases,
                                     T* C,
                                     int64_t* total_rows_before_expert,
                                     int64_t gemm_n,
                                     int64_t gemm_k,
                                     int num_experts,
                                     CutlassGemmConfig gemm_config,
                                     const int multi_processor_count,
                                     cudaStream_t stream,
                                     int* kernel_occupancy = nullptr) {
  if (gemm_config.split_k_style != SplitKStyle::NO_SPLIT_K) {
    PADDLE_FATAL("[MoeGemm] Grouped gemm does not support split-k");
  }

#ifdef PADDLE_CUDA_BF16
  static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value ||
                    cutlass::platform::is_same<T, half>::value ||
                    cutlass::platform::is_same<T, float>::value,
                "Specialized for bfloat16, half, float");
#else
  static_assert(cutlass::platform::is_same<T, half>::value ||
                    cutlass::platform::is_same<T, float>::value,
                "Specialized for half, float");
#endif

  static_assert(
      cutlass::platform::is_same<T, WeightType>::value ||
          cutlass::platform::is_same<WeightType, uint8_t>::value ||
          cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
      "");

  // The cutlass type for the input elements. This is needed to convert to
  // cutlass::half_t if necessary.
  using ElementType_ = typename cutlass::platform::conditional<
      cutlass::platform::is_same<T, half>::value,
      cutlass::half_t,
      T>::type;
#ifdef PADDLE_CUDA_BF16
  using ElementType = typename cutlass::platform::conditional<
      cutlass::platform::is_same<ElementType_, __nv_bfloat16>::value,
      cutlass::bfloat16_t,
      ElementType_>::type;
#else
  using ElementType = ElementType_;
#endif

  using CutlassWeightType_ = typename cutlass::platform::conditional<
      cutlass::platform::is_same<WeightType, half>::value,
      cutlass::half_t,
      WeightType>::type;
#ifdef PADDLE_CUDA_BF16
  using CutlassWeightType = typename cutlass::platform::conditional<
      cutlass::platform::is_same<CutlassWeightType_, __nv_bfloat16>::value,
      cutlass::bfloat16_t,
      CutlassWeightType_>::type;
#else
  using CutlassWeightType = CutlassWeightType_;
#endif

  // We need separate config for each architecture since we will target
  // different tensorcore instructions. For float, we do not target TCs.
  using MixedGemmArchTraits = cutlass::gemm::kernel::
      MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
  using ElementAccumulator = typename MixedGemmArchTraits::AccType;

  using EpilogueOp = typename Epilogue<ElementType,
                                       MixedGemmArchTraits::ElementsPerAccessC,
                                       ElementAccumulator,
                                       EpilogueTag>::Op;

  // Finally, set up the kernel.
  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      ElementType,
      cutlass::layout::RowMajor,
      cutlass::ComplexTransform::kNone,
      MixedGemmArchTraits::ElementsPerAccessA,
      CutlassWeightType,
      typename MixedGemmArchTraits::LayoutB,
      cutlass::ComplexTransform::kNone,
      MixedGemmArchTraits::ElementsPerAccessB,
      ElementType,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      typename MixedGemmArchTraits::OperatorClass,
      arch,
      ThreadblockShape,
      WarpShape,
      typename MixedGemmArchTraits::InstructionShape,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      Stages,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
      typename MixedGemmArchTraits::Operator>::GemmKernel;

  using GemmKernel =
      cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma,
                                       typename GemmKernel_::Epilogue,
                                       typename GemmKernel_::ThreadblockSwizzle,
                                       arch,  // Ensure top level arch is used
                                              // for dispatch
                                       GemmKernel_::kGroupScheduleMode>;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  if (kernel_occupancy != nullptr) {
    *kernel_occupancy = compute_occupancy_for_kernel<GemmKernel>();
    return;
  }
  int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());

  if (occupancy == 0) {
    PADDLE_FATAL(
        "[MoE Runner] GPU lacks the shared memory resources to run "
        "GroupedGEMM kernel");
  }
  const int threadblock_count = multi_processor_count * occupancy;

  typename EpilogueOp::Params epilogue_op(ElementAccumulator(1.f),
                                          ElementAccumulator(0.f));

  typename GemmGrouped::Arguments args(
      num_experts,
      threadblock_count,
      epilogue_op,
      reinterpret_cast<const ElementType*>(A),
      reinterpret_cast<const CutlassWeightType*>(B),
      reinterpret_cast<const ElementType*>(weight_scales),
      reinterpret_cast<const ElementType*>(biases),
      reinterpret_cast<ElementType*>(C),
      total_rows_before_expert,
      gemm_n,
      gemm_k);

  GemmGrouped gemm;

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg = "MoEFC kernel will fail for params. Error: " +
                          std::string(cutlassGetStatusString(can_implement));
    PADDLE_FATAL("[MoE Runner] " + err_msg);
  }

  auto init_status = gemm.initialize(args);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to initialize cutlass variable batched gemm. Error: " +
        std::string(cutlassGetStatusString(init_status));
    PADDLE_FATAL("[MoE Runner] " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to run cutlass variable batched gemm. Error: " +
        std::string(cutlassGetStatusString(run_status));
    PADDLE_FATAL("[MoE Runner] " + err_msg);
  }
}

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages,
          typename Enable = void>
struct dispatch_stages {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const T* weight_scales,
                       const T* biases,
                       T* C,
                       int64_t* total_rows_before_expert,
                       int64_t gemm_n,
                       int64_t gemm_k,
                       int num_experts,
                       CutlassGemmConfig gemm_config,
                       int multi_processor_count,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    std::string err_msg = "Cutlass fpA_intB gemm. Not instantiates for arch " +
                          std::to_string(arch::kMinComputeCapability) +
                          " with stages set to " + std::to_string(Stages);
    PADDLE_FATAL("[dispatch_stages::dispatch] " + err_msg);
  }
};

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape>
struct dispatch_stages<T,
                       WeightType,
                       arch,
                       EpilogueTag,
                       ThreadblockShape,
                       WarpShape,
                       2> {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const T* weight_scales,
                       const T* biases,
                       T* C,
                       int64_t* total_rows_before_expert,
                       int64_t gemm_n,
                       int64_t gemm_k,
                       int num_experts,
                       CutlassGemmConfig gemm_config,
                       int multi_processor_count,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    generic_moe_gemm_kernelLauncher<T,
                                    WeightType,
                                    arch,
                                    EpilogueTag,
                                    ThreadblockShape,
                                    WarpShape,
                                    2>(A,
                                       B,
                                       weight_scales,
                                       biases,
                                       C,
                                       total_rows_before_expert,
                                       gemm_n,
                                       gemm_k,
                                       num_experts,
                                       gemm_config,
                                       multi_processor_count,
                                       stream,
                                       occupancy);
  }
};

template <typename T,
          typename WeightType,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
struct dispatch_stages<T,
                       WeightType,
                       cutlass::arch::Sm80,
                       EpilogueTag,
                       ThreadblockShape,
                       WarpShape,
                       Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const T* weight_scales,
                       const T* biases,
                       T* C,
                       int64_t* total_rows_before_expert,
                       int64_t gemm_n,
                       int64_t gemm_k,
                       int num_experts,
                       CutlassGemmConfig gemm_config,
                       int multi_processor_count,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    generic_moe_gemm_kernelLauncher<T,
                                    WeightType,
                                    cutlass::arch::Sm80,
                                    EpilogueTag,
                                    ThreadblockShape,
                                    WarpShape,
                                    Stages>(A,
                                            B,
                                            weight_scales,
                                            biases,
                                            C,
                                            total_rows_before_expert,
                                            gemm_n,
                                            gemm_k,
                                            num_experts,
                                            gemm_config,
                                            multi_processor_count,
                                            stream,
                                            occupancy);
  }
};

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape>
void dispatch_gemm_config(const T* A,
                          const WeightType* B,
                          const T* weight_scales,
                          const T* biases,
                          T* C,
                          int64_t* total_rows_before_expert,
                          int64_t gemm_n,
                          int64_t gemm_k,
                          int num_experts,
                          CutlassGemmConfig gemm_config,
                          int multi_processor_count,
                          cudaStream_t stream,
                          int* occupancy = nullptr) {
  switch (gemm_config.stages) {
    case 2:
      using DispatcherStages2 = dispatch_stages<T,
                                                WeightType,
                                                arch,
                                                EpilogueTag,
                                                ThreadblockShape,
                                                WarpShape,
                                                2>;
      DispatcherStages2::dispatch(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  total_rows_before_expert,
                                  gemm_n,
                                  gemm_k,
                                  num_experts,
                                  gemm_config,
                                  multi_processor_count,
                                  stream,
                                  occupancy);
      break;
    case 3:
      using DispatcherStages3 = dispatch_stages<T,
                                                WeightType,
                                                arch,
                                                EpilogueTag,
                                                ThreadblockShape,
                                                WarpShape,
                                                3>;
      DispatcherStages3::dispatch(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  total_rows_before_expert,
                                  gemm_n,
                                  gemm_k,
                                  num_experts,
                                  gemm_config,
                                  multi_processor_count,
                                  stream,
                                  occupancy);
      break;
    case 4:
      using DispatcherStages4 = dispatch_stages<T,
                                                WeightType,
                                                arch,
                                                EpilogueTag,
                                                ThreadblockShape,
                                                WarpShape,
                                                4>;
      DispatcherStages4::dispatch(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  total_rows_before_expert,
                                  gemm_n,
                                  gemm_k,
                                  num_experts,
                                  gemm_config,
                                  multi_processor_count,
                                  stream,
                                  occupancy);
      break;
    case 5:
      using DispatcherStages5 = dispatch_stages<T,
                                                WeightType,
                                                arch,
                                                EpilogueTag,
                                                ThreadblockShape,
                                                WarpShape,
                                                5>;
      DispatcherStages5::dispatch(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  total_rows_before_expert,
                                  gemm_n,
                                  gemm_k,
                                  num_experts,
                                  gemm_config,
                                  multi_processor_count,
                                  stream,
                                  occupancy);
      break;
    default:
      std::string err_msg = "dispatch_gemm_config does not support stages " +
                            std::to_string(gemm_config.stages);
      PADDLE_FATAL("[MoE][dispatch_gemm_config] " + err_msg);
      break;
  }
}

// This overload will handle tensorop gemms. It is disabled via SFINAE for fp32.
// This overload is only enabled when T == WeightType.
template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename std::enable_if<!std::is_same<T, float>::value &&
                                  std::is_same<T, WeightType>::value>::type* =
              nullptr>
void dispatch_moe_gemm_to_cutlass(const T* A,
                                  const WeightType* B,
                                  const T* weight_scales,
                                  const T* biases,
                                  T* C,
                                  int64_t* total_rows_before_expert,
                                  int64_t total_rows,
                                  int64_t gemm_n,
                                  int64_t gemm_k,
                                  int num_experts,
                                  CutlassGemmConfig gemm_config,
                                  int sm_version,
                                  int multi_processor_count,
                                  cudaStream_t stream,
                                  int* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<32, 128, 64>,
                           cutlass::gemm::GemmShape<32, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<64, 128, 64>,
                           cutlass::gemm::GemmShape<32, 64, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<128, 128, 64>,
                           cutlass::gemm::GemmShape<64, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<128, 256, 64>,
                           cutlass::gemm::GemmShape<64, 64, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::Undefined:
      PADDLE_FATAL("[dispatch_moe_gemm_to_cutlass] gemm config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      PADDLE_FATAL(
          "[dispatch_moe_gemm_to_cutlass] gemm config should have "
          "already been set by heuristic.");
      break;
    default:
      PADDLE_FATAL(
          "[dispatch_moe_gemm_to_cutlass] Config is invalid for same "
          "type MoE tensorop GEMM.");
      break;
  }
}

// Tensorop GEMM overload
// Overload for quantize MoE GEMMs. We disable some warp configs here since they
// will not be used and we can improve compile time
template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename std::enable_if<!std::is_same<T, float>::value &&
                                  !std::is_same<T, WeightType>::value>::type* =
              nullptr>
void dispatch_moe_gemm_to_cutlass(const T* A,
                                  const WeightType* B,
                                  const T* weight_scales,
                                  const T* biases,
                                  T* C,
                                  int64_t* total_rows_before_expert,
                                  int64_t total_rows,
                                  int64_t gemm_n,
                                  int64_t gemm_k,
                                  int num_experts,
                                  CutlassGemmConfig gemm_config,
                                  int sm_version,
                                  int multi_processor_count,
                                  cudaStream_t stream,
                                  int* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<32, 128, 64>,
                           cutlass::gemm::GemmShape<32, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<32, 128, 64>,
                           cutlass::gemm::GemmShape<32, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<64, 128, 64>,
                           cutlass::gemm::GemmShape<64, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x64x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<64, 128, 64>,
                           cutlass::gemm::GemmShape<64, 64, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<128, 128, 64>,
                           cutlass::gemm::GemmShape<64, 64, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<128, 128, 64>,
                           cutlass::gemm::GemmShape<128, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<128, 256, 64>,
                           cutlass::gemm::GemmShape<64, 64, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::Undefined:
      PADDLE_FATAL("[dispatch_moe_gemm_to_cutlass] gemm config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      PADDLE_FATAL(
          "[dispatch_moe_gemm_to_cutlass] gemm config should have "
          "already been set by heuristic.");
      break;
    default:
      PADDLE_FATAL(
          "[dispatch_moe_gemm_to_cutlass] Config is invalid for "
          "mixed type tensorop GEMM.");
      break;
  }
}

// This overload will handle simt gemms. It is disabled via SFINAE for tensorop.
template <
    typename T,
    typename WeightType,
    typename arch,
    typename EpilogueTag,
    typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
void dispatch_moe_gemm_to_cutlass(const T* A,
                                  const WeightType* B,
                                  const T* weight_scales,
                                  const T* biases,
                                  T* C,
                                  int64_t* total_rows_before_expert,
                                  int64_t total_rows,
                                  int64_t gemm_n,
                                  int64_t gemm_k,
                                  int num_experts,
                                  CutlassGemmConfig gemm_config,
                                  int sm_version,
                                  int multi_processor_count,
                                  cudaStream_t stream,
                                  int* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<128, 128, 8>,
                           cutlass::gemm::GemmShape<64, 64, 8>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::Undefined:
      PADDLE_FATAL(
          "[dispatch_moe_gemm_to_cutlass][SIMT] gemm config "
          "undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      PADDLE_FATAL(
          "[dispatch_moe_gemm_to_cutlass][SIMT] gemm config should "
          "have already been set by heuristic.");
      break;
    default:
      PADDLE_FATAL(
          "[dispatch_moe_gemm_to_cutlass][SIMT] Unsupported config "
          "for float MoE gemm.");
      break;
  }
}

template <typename T, typename WeightType>
MoeGemmRunner<T, WeightType>::MoeGemmRunner() {
  int device{-1};
  check_cuda_error(cudaGetDevice(&device));
  sm_ = getSMVersion();
  check_cuda_error(cudaDeviceGetAttribute(
      &multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::dispatch_to_arch<EpilogueTag>(
    const T* A,
    const WeightType* B,
    const T* weight_scales,
    const T* biases,
    T* C,
    int64_t* total_rows_before_expert,
    int64_t total_rows,
    int64_t gemm_n,
    int64_t gemm_k,
    int num_experts,
    CutlassGemmConfig gemm_config,
    cudaStream_t stream,
    int* occupancy) {
  if (sm_ >= 70 && sm_ < 75) {
    dispatch_moe_gemm_to_cutlass<T,
                                 WeightType,
                                 cutlass::arch::Sm70,
                                 EpilogueTag>(A,
                                              B,
                                              weight_scales,
                                              biases,
                                              C,
                                              total_rows_before_expert,
                                              total_rows,
                                              gemm_n,
                                              gemm_k,
                                              num_experts,
                                              gemm_config,
                                              sm_,
                                              multi_processor_count_,
                                              stream,
                                              occupancy);
  } else if (sm_ >= 75 && sm_ < 80) {
    dispatch_moe_gemm_to_cutlass<T,
                                 WeightType,
                                 cutlass::arch::Sm75,
                                 EpilogueTag>(A,
                                              B,
                                              weight_scales,
                                              biases,
                                              C,
                                              total_rows_before_expert,
                                              total_rows,
                                              gemm_n,
                                              gemm_k,
                                              num_experts,
                                              gemm_config,
                                              sm_,
                                              multi_processor_count_,
                                              stream,
                                              occupancy);
  } else if (sm_ >= 80 && sm_ < 91) {
    dispatch_moe_gemm_to_cutlass<T,
                                 WeightType,
                                 cutlass::arch::Sm80,
                                 EpilogueTag>(A,
                                              B,
                                              weight_scales,
                                              biases,
                                              C,
                                              total_rows_before_expert,
                                              total_rows,
                                              gemm_n,
                                              gemm_k,
                                              num_experts,
                                              gemm_config,
                                              sm_,
                                              multi_processor_count_,
                                              stream,
                                              occupancy);
  } else {
    PADDLE_FATAL("[MoE][GEMM Dispatch] Arch unsupported for MoE GEMM");
  }
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::run_gemm<EpilogueTag>(
    const T* A,
    const WeightType* B,
    const T* weight_scales,
    const T* biases,
    T* C,
    int64_t* total_rows_before_expert,
    int64_t total_rows,
    int64_t gemm_n,
    int64_t gemm_k,
    int num_experts,
    cudaStream_t stream) {
  static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;
  static constexpr bool only_simt_configs = std::is_same<T, float>::value;
  std::vector<CutlassGemmConfig> candidate_configs = get_candidate_configs(
      sm_, -1, is_weight_only, false, only_simt_configs, true);
  static constexpr int warm_time = 5;
  static constexpr int test_time = 10;
  auto& gemmConfigManager = phi::GemmConfigManager::Instance();
  constexpr GemmDataType dtype = getGemmDataType<T>();
  constexpr GemmDataType wdtype = getGemmDataType<WeightType>();
  GemmIDType gemmId{
      gemm_n, gemm_k, GemmType::MOEGEMM, dtype, wdtype, num_experts};
  CutlassGemmConfig chosen_config;
  auto chosen_config_optional =
      gemmConfigManager.getBestConfig(gemmId, total_rows);
  if (chosen_config_optional != std::nullopt) {
    chosen_config = chosen_config_optional.value();
  } else {
    float best_time = std::numeric_limits<float>::max();
    CutlassGemmConfig best_config;
    int profile_total_rows =
        std::min(gemmConfigManager.nextPowerOfTwo(total_rows),
                 gemmConfigManager.getMaxProfileM());
    bool found_one = false;

    for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
      for (int i = 0; i < warm_time; i++) {
        dispatch_to_arch<EpilogueTag>(A,
                                      B,
                                      weight_scales,
                                      biases,
                                      C,
                                      total_rows_before_expert,
                                      total_rows,
                                      gemm_n,
                                      gemm_k,
                                      num_experts,
                                      candidate_configs[ii],
                                      stream);
      }
      cudaEvent_t start;
      cudaEvent_t stop;
      check_cuda_error(cudaEventCreate(&start));
      check_cuda_error(cudaEventCreate(&stop));
      check_cuda_error(cudaStreamSynchronize(stream));
      check_cuda_error(cudaEventRecord(start, stream));
      for (int i = 0; i < test_time; i++) {
        dispatch_to_arch<EpilogueTag>(A,
                                      B,
                                      weight_scales,
                                      biases,
                                      C,
                                      total_rows_before_expert,
                                      total_rows,
                                      gemm_n,
                                      gemm_k,
                                      num_experts,
                                      candidate_configs[ii],
                                      stream);
      }
      check_cuda_error(cudaEventRecord(stop, stream));
      check_cuda_error(cudaEventSynchronize(stop));
      found_one = true;
      float elapsed;
      check_cuda_error(cudaEventElapsedTime(&elapsed, start, stop));
      check_cuda_error(cudaEventDestroy(start));
      check_cuda_error(cudaEventDestroy(stop));
      if (elapsed < best_time) {
        best_time = elapsed;
        best_config = candidate_configs[ii];
      }
      VLOG(4) << "profile_total_rows" << profile_total_rows;
      VLOG(4) << "candidate_config tile_config"
              << static_cast<int>(candidate_configs[ii].tile_config);
      VLOG(4) << "candidate_config split_k_style"
              << static_cast<int>(candidate_configs[ii].split_k_style);
      VLOG(4) << "candidate_config split_k_factor "
              << candidate_configs[ii].split_k_factor;
      VLOG(4) << "candidate_config stages " << candidate_configs[ii].stages;
      VLOG(4) << "elapsed time: " << elapsed;
      VLOG(4) << "best_time: " << best_time;
    }
    if (found_one) {
      gemmConfigManager.addBestConfig(gemmId, profile_total_rows, best_config);
      chosen_config = best_config;
    }
  }

  VLOG(4) << "chosen_config tile_config"
          << static_cast<int>(chosen_config.tile_config);
  VLOG(4) << "chosen_config split_k_style"
          << static_cast<int>(chosen_config.split_k_style);
  VLOG(4) << "chosen_config split_k_factor " << chosen_config.split_k_factor;
  VLOG(4) << "chosen_config stages " << chosen_config.stages;
  VLOG(4) << "total_rows  " << total_rows << "gemm_n  " << gemm_n << "gemm_k  "
          << gemm_k;

  dispatch_to_arch<EpilogueTag>(A,
                                B,
                                weight_scales,
                                biases,
                                C,
                                total_rows_before_expert,
                                total_rows,
                                gemm_n,
                                gemm_k,
                                num_experts,
                                chosen_config,
                                stream);
}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moe_gemm_bias_act(
    const T* A,
    const WeightType* B,
    const T* weight_scales,
    const T* biases,
    T* C,
    int64_t* total_rows_before_expert,
    int64_t total_rows,
    int64_t gemm_n,
    int64_t gemm_k,
    int num_experts,
    std::string activation_type,
    cudaStream_t stream) {
  if (activation_type == "none") {
    if (biases) {
      run_gemm<EpilogueOpBias>(A,
                               B,
                               weight_scales,
                               biases,
                               C,
                               total_rows_before_expert,
                               total_rows,
                               gemm_n,
                               gemm_k,
                               num_experts,
                               stream);
    } else {
      run_gemm<EpilogueOpNoBias>(A,
                                 B,
                                 weight_scales,
                                 nullptr,
                                 C,
                                 total_rows_before_expert,
                                 total_rows,
                                 gemm_n,
                                 gemm_k,
                                 num_experts,
                                 stream);
    }
  }
}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moe_gemm(const T* A,
                                            const WeightType* B,
                                            const T* weight_scales,
                                            T* C,
                                            int64_t* total_rows_before_expert,
                                            int64_t total_rows,
                                            int64_t gemm_n,
                                            int64_t gemm_k,
                                            int num_experts,
                                            cudaStream_t stream) {
  run_gemm<EpilogueOpNoBias>(A,
                             B,
                             weight_scales,
                             nullptr,
                             C,
                             total_rows_before_expert,
                             total_rows,
                             gemm_n,
                             gemm_k,
                             num_experts,
                             stream);
}

}  // namespace phi
