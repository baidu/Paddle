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

#include <glog/logging.h>
#include <iostream>

#include "./cutlass_kernels/fp8_fp8_dual_gemm_scale_bias_act.h"
#include "fp8_common.h"  // NOLINT

#include "paddle/phi/backends/dynload/fp8_gemm_fused.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

template <typename InputType, typename Context>
void fp8_fp8_fp8_dual_gemm(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y0,
    const DenseTensor& y1,
    const paddle::optional<DenseTensor>& bias0,
    const paddle::optional<DenseTensor>& bias1,
    const bool trans_x,
    const bool trans_y,
    const float scale0,     // only support per-tensor quantization
    const float scale1,     // only support per-tensor quantization
    const float scale_out,  // only support per-tensor quantization
    const std::string& activation_type,
    DenseTensor* out) {
  static_assert(std::is_same<Context, phi::GPUContext>::value,
                "fp8_fp8_fp8_dual_gemm_fused must be in GPU");
  VLOG(3) << "fp8_fp8_fp8_dual_gemm_fused of cutlass start run: ";

  dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(out);
  auto place = dev_ctx.GetPlace();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(dev_ctx.stream());
  int64_t device_id = place.GetDeviceId();
  int sm_version = backends::gpu::GetGPUComputeCapability(device_id);

  int rank = x.dims().size();
  int M = 0;
  int K = 0;
  int N = 0;
  int lda = x.dims()[rank - 1];
  int ldb = y0.dims()[rank - 1];
  int ldd = out->dims()[rank - 1];
  if (!trans_x) {
    M = x.dims()[rank - 2];
    K = x.dims()[rank - 1];

  } else {
    M = x.dims()[rank - 1];
    K = x.dims()[rank - 2];
  }

  if (!trans_y) {
    N = y0.dims()[rank - 1];
  } else {
    N = y0.dims()[rank - 2];
  }

  int batch_count = 1;
  for (size_t i = 0; i < rank - 2; ++i) {
    batch_count *= x.dims()[i];
  }

  void* dlhandler = phi::dynload::GetFP8FP8GemmFusedHandle();
  if (!dlhandler) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The library of fp8 phi kernels has not been compiled. Please run the "
        "compile.sh script according to README to compile the cutlass kernels "
        "library"));
  }

  std::string input_dtype =
      (x.dtype() == phi::DataType::FLOAT8_E4M3FN) ? "e4m3" : "e5m2";
  std::string output_dtype = "e4m3";

  std::string isbias;
  std::string bias_dtype;
  void* bias_data0 = nullptr;
  void* bias_data1 = nullptr;
  std::vector<int64_t> bias_dims0{};
  std::vector<int64_t> bias_dims1{};
  if (bias0 && bias1) {
    isbias = "bias_";
    bias_dims0 = common::vectorize(bias0.get().dims());
    bias_dims1 = common::vectorize(bias1.get().dims());
    if (bias0.get().dtype() == phi::DataType::FLOAT16) {
      bias_dtype = "fp16_";
      bias_data0 = reinterpret_cast<void*>(const_cast<phi::dtype::float16*>(
          bias0.get().data<phi::dtype::float16>()));
      bias_data1 = reinterpret_cast<void*>(const_cast<phi::dtype::float16*>(
          bias1.get().data<phi::dtype::float16>()));
    } else {
      bias_dtype = "bf16_";
      bias_data0 = reinterpret_cast<void*>(const_cast<phi::dtype::bfloat16*>(
          bias0.get().data<phi::dtype::bfloat16>()));
      bias_data1 = reinterpret_cast<void*>(const_cast<phi::dtype::bfloat16*>(
          bias1.get().data<phi::dtype::bfloat16>()));
    }
  }
  std::string act = (activation_type == "") ? "swiglu" : activation_type;

  std::string gemm_config =
      input_dtype + "_" + output_dtype + "_" + isbias + bias_dtype + act;

  DualGemmEpilogueAllParams params = {
      reinterpret_cast<const void*>(x.data<InputType>()),
      reinterpret_cast<const void*>(y0.data<InputType>()),
      reinterpret_cast<const void*>(y1.data<InputType>()),
      reinterpret_cast<void*>(out->data<phi::dtype::float8_e4m3fn>()),
      scale0,
      scale1,
      scale_out,
      M,
      N,
      K,
      lda,
      ldb,
      ldd,
      batch_count,
      place,
      stream,
      sm_version,
      bias_data0,
      bias_data1,
      bias_dims0,
      bias_dims1,
      gemm_config};
  func1 fp8_gemm_func =
      (func1)(dlsym(dlhandler, "fp8_fp8_dual_gemm_scale_bias_act"));
  fp8_gemm_func(params);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fp8_fp8_fp8_dual_gemm_fused,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::fp8_fp8_fp8_dual_gemm,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2) {}
