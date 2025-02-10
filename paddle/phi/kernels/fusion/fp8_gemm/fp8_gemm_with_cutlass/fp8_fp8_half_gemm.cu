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

#include "./cutlass_kernels/fp8_fp8_gemm_scale_bias_act.h"
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
void fp8_fp8_half_gemm(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const paddle::optional<DenseTensor>& bias,
    const bool trans_x,
    const bool trans_y,
    const float scale,  // only support per-tensor quantization
    const std::string& output_dtype,
    const std::string& activation_type,
    DenseTensor* out) {
  static_assert(std::is_same<Context, phi::GPUContext>::value,
                "fp8_fp8_gemm must be in GPU");

  VLOG(3) << "fp8_fp8_half_gemm_fused of cutlass start run: ";

  void* out_ptr = nullptr;
  if (out->dtype() == phi::DataType::BFLOAT16) {
    dev_ctx.template Alloc<phi::dtype::bfloat16>(out);
    out_ptr = reinterpret_cast<void*>(out->data<phi::dtype::bfloat16>());
  } else if (out->dtype() == phi::DataType::FLOAT16) {
    dev_ctx.template Alloc<phi::dtype::float16>(out);
    out_ptr = reinterpret_cast<void*>(out->data<phi::dtype::float16>());
  } else {
    PADDLE_THROW(phi::errors::Fatal(
        "fp8_fp8_half_gemm_fused only support bfloat16 and float16 output"));
  }
  auto place = dev_ctx.GetPlace();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(dev_ctx.stream());
  int64_t device_id = place.GetDeviceId();
  int sm_version = backends::gpu::GetGPUComputeCapability(device_id);

  int rank = x.dims().size();
  int M = 0;
  int K = 0;
  int N = 0;
  int lda = x.dims()[rank - 1];
  int ldb = y.dims()[rank - 1];
  int ldd = out->dims()[rank - 1];
  if (!trans_x) {
    M = x.dims()[rank - 2];
    K = x.dims()[rank - 1];

  } else {
    M = x.dims()[rank - 1];
    K = x.dims()[rank - 2];
  }

  if (!trans_y) {
    N = y.dims()[rank - 1];
  } else {
    N = y.dims()[rank - 2];
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
  std::string cutlass_output_dtype = "";
  if (output_dtype == "bfloat16") {
    cutlass_output_dtype = std::string("bf16");
  } else if (output_dtype == "float16") {
    cutlass_output_dtype = std::string("fp16");
  } else {
    PADDLE_THROW(phi::errors::Fatal(
        "fp8_fp8_half_gemm_fused only support bfloat16 and float16 output"));
  }
  std::string isbias = bias ? "bias_" : "";
  std::string act = (activation_type == "" || activation_type == "identity")
                        ? "identity"
                        : activation_type;

  std::string gemm_config =
      input_dtype + "_" + cutlass_output_dtype + "_" + isbias + act;

  void* bias_data = nullptr;
  std::vector<int64_t> bias_dims{};
  if (bias) {
    bias_dims = common::vectorize(bias.get().dims());
    if (output_dtype == "bfloat16") {
      bias_data = reinterpret_cast<void*>(const_cast<phi::dtype::bfloat16*>(
          bias.get().data<phi::dtype::bfloat16>()));
    } else {
      bias_data = reinterpret_cast<void*>(const_cast<phi::dtype::float16*>(
          bias.get().data<phi::dtype::float16>()));
    }
  }
  GemmEpilogueAllParams params = {
      reinterpret_cast<const void*>(x.data<InputType>()),
      reinterpret_cast<const void*>(y.data<InputType>()),
      out_ptr,
      scale,
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
      0.01,  // for leaky_relu
      bias_data,
      bias_dims,
      gemm_config};
  func fp8_gemm_func = (func)(dlsym(dlhandler, "fp8_fp8_gemm_scale_bias_act"));
  fp8_gemm_func(params);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fp8_fp8_half_gemm_fused,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::fp8_fp8_half_gemm,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2) {}
