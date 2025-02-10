// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <cub/cub.cuh>  // NOLINT
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/preln_residual_bias_plugin.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"
#include "paddle/phi/kernels/funcs/math/bert_encoder_functor.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_common.h"
#include "paddle/phi/kernels/fusion/gpu/fused_layernorm_residual_dropout_bias.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

inline int getSMVersion() {
  const int device = phi::backends::gpu::GetCurrentDeviceId();
  const phi::gpuDeviceProp prop =
      phi::backends::gpu::GetDeviceProperties(device);
  return prop.major * 10 + prop.minor;
}

#ifdef TRT_PLUGIN_FP16_AVAILABLE
#define FINAL_MASK 0xffffffff

template <int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt2(
    half2 *normed_output,
    half2 *output,
    const half2 *__restrict bias,
    const half2 *__restrict src,
    const half2 *__restrict residual,
    const half2 *__restrict gamma,
    const half2 *__restrict beta,
    int m,
    int n,
    float epsilon) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  __shared__ float s_mean;
  __shared__ float s_variance;
  float x_sum = 0.0f;
  float x2_sum = 0.0f;
  const int b_offset = blockIdx.x * n;

#pragma unroll UNROLL_FACTOR
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int index = b_offset + i;
    float val_1 = 0.0f;
    float val_2 = 0.0f;
    half2 tmp;

    if (bias) {
      tmp = __ldg(&bias[i]);
      val_1 += static_cast<float>(tmp.x);
      val_2 += static_cast<float>(tmp.y);
    }
    {
      tmp = __ldg(&residual[index]);
      val_1 += static_cast<float>(tmp.x);
      val_2 += static_cast<float>(tmp.y);
    }
    {
      tmp = __ldg(&src[index]);
      val_1 += static_cast<float>(tmp.x);
      val_2 += static_cast<float>(tmp.y);
    }
    tmp.x = __float2half_rn(val_1);
    tmp.y = __float2half_rn(val_2);
    output[index] = tmp;
    x_sum += val_1 + val_2;
    x2_sum += val_1 * val_1 + val_2 * val_2;
  }
  float sums[2];
  sums[0] = x_sum;
  sums[1] = x2_sum;
  phi::funcs::BlockReduceSumV2<float, 2>(sums);

  if (threadIdx.x == 0) {
    s_mean = sums[0] / n / 2;
    s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + epsilon);
  }
  __syncthreads();

  half2 mean_2 = __float2half2_rn(s_mean);
  half2 var_2 = __float2half2_rn(s_variance);

#pragma unroll UNROLL_FACTOR
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int index = b_offset + i;
    half2 val = __hmul2(__hmul2(__hsub2(output[index], mean_2), var_2),
                        __ldg(&gamma[i]));
    if (beta) {
      val = __hadd2(val, __ldg(&beta[i]));
    }
    normed_output[index] = val;
  }
#endif
}

#define HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(UNROLL_FACTOR)                \
  generalAddBiasResidualLayerNormOpt2<UNROLL_FACTOR>                         \
      <<<rows, block, 0, stream>>>(reinterpret_cast<half2 *>(layernorm_dst), \
                                   reinterpret_cast<half2 *>(dst),           \
                                   (const half2 *)bias,                      \
                                   (const half2 *)input2,                    \
                                   (const half2 *)input1,                    \
                                   (const half2 *)fp16_scale_gpu_,           \
                                   (const half2 *)fp16_bias_gpu_,            \
                                   rows,                                     \
                                   half_n,                                   \
                                   epsilon);

#endif

using half = phi::dtype::float16;

#if IS_TRT_VERSION_GE(6000)
int PrelnResidualBiasPluginDynamic::initialize() TRT_NOEXCEPT {
  cudaMalloc(&bias_gpu_, sizeof(float) * bias_size_);
  cudaMemcpy(bias_gpu_,
             bias_.data(),
             bias_size_ * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&scale_gpu_, sizeof(float) * scale_size_);
  cudaMemcpy(scale_gpu_,
             scale_.data(),
             scale_size_ * sizeof(float),
             cudaMemcpyHostToDevice);
  if (with_fp16_) {
    cudaMalloc(&fp16_bias_gpu_, sizeof(half) * bias_size_);
    cudaMemcpy(fp16_bias_gpu_,
               fp16_bias_.data(),
               bias_size_ * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMalloc(&fp16_scale_gpu_, sizeof(half) * scale_size_);
    cudaMemcpy(fp16_scale_gpu_,
               fp16_scale_.data(),
               scale_size_ * sizeof(half),
               cudaMemcpyHostToDevice);
  }
  if (ele_bias_size_ > 0) {
    if (with_fp16_) {
      cudaMalloc(&ele_bias_gpu_, sizeof(half) * ele_bias_size_);
      cudaMemcpy(ele_bias_gpu_,
                 fp16_ele_bias_.data(),
                 ele_bias_size_ * sizeof(half),
                 cudaMemcpyHostToDevice);
    } else {
      cudaMalloc(&ele_bias_gpu_, sizeof(float) * ele_bias_size_);
      cudaMemcpy(ele_bias_gpu_,
                 fp32_ele_bias_.data(),
                 ele_bias_size_ * sizeof(float),
                 cudaMemcpyHostToDevice);
    }
  } else {
    ele_bias_gpu_ = nullptr;
  }

  return 0;
}

void PrelnResidualBiasPluginDynamic::terminate() TRT_NOEXCEPT {
  if (bias_gpu_) {
    cudaFree(bias_gpu_);
    bias_gpu_ = nullptr;
  }
  if (fp16_bias_gpu_) {
    cudaFree(fp16_bias_gpu_);
    fp16_bias_gpu_ = nullptr;
  }
  if (scale_gpu_) {
    cudaFree(scale_gpu_);
    scale_gpu_ = nullptr;
  }
  if (fp16_scale_gpu_) {
    cudaFree(fp16_scale_gpu_);
    fp16_scale_gpu_ = nullptr;
  }
  if (ele_bias_gpu_) {
    cudaFree(ele_bias_gpu_);
    ele_bias_gpu_ = nullptr;
  }
}

nvinfer1::IPluginV2DynamicExt *PrelnResidualBiasPluginDynamic::clone() const
    TRT_NOEXCEPT {
  PrelnResidualBiasPluginDynamic *ptr = nullptr;
  if (with_fp16_) {
    ptr = new PrelnResidualBiasPluginDynamic(bias_.data(),
                                             scale_.data(),
                                             fp16_ele_bias_.data(),
                                             bias_size_,
                                             scale_size_,
                                             ele_bias_size_,
                                             eps_,
                                             with_fp16_);
  } else {
    ptr = new PrelnResidualBiasPluginDynamic(bias_.data(),
                                             scale_.data(),
                                             fp32_ele_bias_.data(),
                                             bias_size_,
                                             scale_size_,
                                             ele_bias_size_,
                                             eps_,
                                             with_fp16_);
  }

  ptr->bias_gpu_ = bias_gpu_;
  ptr->fp16_bias_gpu_ = fp16_bias_gpu_;
  ptr->scale_gpu_ = scale_gpu_;
  ptr->fp16_scale_gpu_ = fp16_scale_gpu_;
  ptr->ele_bias_gpu_ = ele_bias_gpu_;
  return ptr;
}

const char *PrelnResidualBiasPluginDynamic::getPluginType() const TRT_NOEXCEPT {
  return "preln_residual_bias_plugin_dynamic";
}

int PrelnResidualBiasPluginDynamic::getNbOutputs() const TRT_NOEXCEPT {
  return 2;
}

size_t PrelnResidualBiasPluginDynamic::getSerializationSize() const
    TRT_NOEXCEPT {
  size_t ser_size = SerializedSize(bias_) + SerializedSize(fp16_bias_) +
                    SerializedSize(scale_) + SerializedSize(fp16_scale_) +
                    SerializedSize(fp32_ele_bias_) +
                    SerializedSize(fp16_ele_bias_) +
                    SerializedSize(bias_size_) + SerializedSize(scale_size_) +
                    SerializedSize(ele_bias_size_) + SerializedSize(eps_) +
                    SerializedSize(with_fp16_);
  return ser_size;
}
void PrelnResidualBiasPluginDynamic::serialize(void *buffer) const
    TRT_NOEXCEPT {
  SerializeValue(&buffer, bias_);
  SerializeValue(&buffer, fp16_bias_);
  SerializeValue(&buffer, scale_);
  SerializeValue(&buffer, fp16_scale_);
  SerializeValue(&buffer, fp32_ele_bias_);
  SerializeValue(&buffer, fp16_ele_bias_);
  SerializeValue(&buffer, bias_size_);
  SerializeValue(&buffer, scale_size_);
  SerializeValue(&buffer, ele_bias_size_);
  SerializeValue(&buffer, eps_);
  SerializeValue(&buffer, with_fp16_);
}

nvinfer1::DimsExprs PrelnResidualBiasPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  if (output_index < 2) {
    return inputs[0];
  } else {  // moving mean and var
    nvinfer1::DimsExprs ret;
    ret.nbDims = 1;
    ret.d[0] = inputs[0].d[2];
    return ret;
  }
}

bool PrelnResidualBiasPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of swish plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVAILABLE
      return (in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
#else
      PADDLE_THROW(
          common::errors::Fatal("TRT plugin supported FP16 is not available "
                                "while with_fp16 is set true."));
#endif
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];

  if (pos == 1) {
    return in.type == prev.type && in.format == prev.format;
  }

  // output
  return in.type == prev.type && in.format == prev.format;
}

void PrelnResidualBiasPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nb_outputs) TRT_NOEXCEPT {}

size_t PrelnResidualBiasPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs,
    int nb_inputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nb_outputs) const TRT_NOEXCEPT {
  return 0;
}

nvinfer1::DataType PrelnResidualBiasPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

void PrelnResidualBiasPluginDynamic::destroy() TRT_NOEXCEPT { delete this; }

int PrelnResidualBiasPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  int hidden;
  int rows_temp;
  int cols_temp;
  if (input_dims.nbDims == 3) {
    hidden = input_dims.d[2];
    rows_temp = input_dims.d[0] * input_dims.d[1];
    cols_temp = input_dims.d[2];
  } else if (input_dims.nbDims == 4) {
    hidden = input_dims.d[3];
    rows_temp = input_dims.d[0] * input_dims.d[1] * input_dims.d[2];
    cols_temp = input_dims.d[3];
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "fused_bias_dropout_residual_layer_norm op only support 3-D or 4-D "
        "input, but get `%d`-D.",
        input_dims.nbDims));
  }
  const size_t rows = static_cast<size_t>(rows_temp);  // batch * seq_length
  const size_t cols = static_cast<size_t>(cols_temp);

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. PrelnResidualBias-->fp32";
    const float *input1 = static_cast<const float *>(inputs[0]);
    const float *input2 = static_cast<const float *>(inputs[1]);

    uint64_t seed = 0;
    const float dropout_prob = 0.;
    const bool is_upscale_in_train = false;
    const bool is_test = true;
    const uint64_t increment = 0;
    const float epsilon = eps_;
    const float *src = input2;
    const float *residual = input1;
    const float *bias = static_cast<float *>(ele_bias_gpu_);
    const float *scale = scale_gpu_;
    const float *layernorm_bias = bias_gpu_;
    uint8_t *mask_data = nullptr;
    float *dst = static_cast<float *>(outputs[1]);
    float *layernorm_dst = static_cast<float *>(outputs[0]);
    float *mean = nullptr;
    float *var = nullptr;
    const int VecSize = 8;
    phi::fusion::FusedLayernormResidualDropoutBiasFunctor<float,
                                                          uint8_t,
                                                          VecSize,
                                                          float,
                                                          false>()(
        rows,
        cols,
        seed,
        dropout_prob,
        is_upscale_in_train,
        is_test,
        increment,
        epsilon,
        src,
        residual,
        bias,
        scale,
        layernorm_bias,
        mask_data,
        dst,
        layernorm_dst,
        mean,
        var,
        stream);

  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef TRT_PLUGIN_FP16_AVAILABLE
    VLOG(1) << "TRT Plugin DataType selected. PrelnResidualBias-->fp16";
    const half *input1 = static_cast<const half *>(inputs[0]);
    const half *input2 = static_cast<const half *>(inputs[1]);

    uint64_t seed = 0;
    const float dropout_prob = 0.;
    const bool is_upscale_in_train = false;
    const bool is_test = true;
    const uint64_t increment = 0;
    const float epsilon = eps_;
    const half *src = input2;
    const half *residual = input1;
    const half *bias = static_cast<half *>(ele_bias_gpu_);
    const float *scale = scale_gpu_;
    const float *layernorm_bias = bias_gpu_;
    uint8_t *mask_data = nullptr;
    half *dst = static_cast<half *>(outputs[1]);
    half *layernorm_dst = static_cast<half *>(outputs[0]);
    float *mean = nullptr;
    float *var = nullptr;
    const int VecSize = 8;

    int sm = getSMVersion();
    // sm >= 60 to support _ldg
    if (sm >= 60) {
      // if hidden is even, use half2 kernel generalAddBiasResidualLayerNormOpt2
      if (hidden % 2 == 0) {
        int half_n = hidden / 2;
        int half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(std::min(half_n_32, 512));
        int rolls_per_thread = half_n / block.x;
        int unroll_factor = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
          unroll_factor /= 2;
        }
        switch (unroll_factor) {
          case 1:
            HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(1);
            break;
          case 2:
            HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(2);
            break;
          case 4:
            HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(4);
            break;
          case 8:
            HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(8);
            break;
          default:
            PADDLE_THROW(common::errors::Fatal(
                "Invalid UNROLL_FACTOR in preln_residual_bias trt plugin."));
        }
      } else {
        phi::fusion::FusedLayernormResidualDropoutBiasFunctor<half,
                                                              uint8_t,
                                                              VecSize,
                                                              float,
                                                              false>()(
            rows,
            cols,
            seed,
            dropout_prob,
            is_upscale_in_train,
            is_test,
            increment,
            epsilon,
            src,
            residual,
            bias,
            scale,
            layernorm_bias,
            mask_data,
            dst,
            layernorm_dst,
            mean,
            var,
            stream);
      }
    } else {
      // if sm < 60, use FusedLayernormResidualDropoutBiasFunctor only
      phi::fusion::FusedLayernormResidualDropoutBiasFunctor<half,
                                                            uint8_t,
                                                            VecSize,
                                                            float,
                                                            false>()(
          rows,
          cols,
          seed,
          dropout_prob,
          is_upscale_in_train,
          is_test,
          increment,
          epsilon,
          src,
          residual,
          bias,
          scale,
          layernorm_bias,
          mask_data,
          dst,
          layernorm_dst,
          mean,
          var,
          stream);
    }
#else
    PADDLE_THROW(common::errors::Fatal(
        "The Ernie(Bert) tensorRT plugin should be "
        "complied with CUDA version >= 10.0 when running with fp16. "
        "Please recompile it or try to use fp32 by set "
        "config.SetTRTDynamicShapeInfo(min_input_shape, "
        "max_input_shape, opt_input_shape, true"));
#endif
  } else {
    PADDLE_THROW(
        common::errors::Fatal("The PrelnResidualBias TRT Plugin's input type "
                              "should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

const char *PrelnResidualBiasPluginDynamicCreator::getPluginName() const
    TRT_NOEXCEPT {
  return "preln_residual_bias_plugin_dynamic";
}

const char *PrelnResidualBiasPluginDynamicCreator::getPluginVersion() const
    TRT_NOEXCEPT {
  return "1";
}

nvinfer1::IPluginV2 *PrelnResidualBiasPluginDynamicCreator::deserializePlugin(
    const char *name,
    const void *serial_data,
    size_t serial_length) TRT_NOEXCEPT {
  return new PrelnResidualBiasPluginDynamic(serial_data, serial_length);
}

int PIRPrelnResidualBiasPluginDynamic::initialize() TRT_NOEXCEPT {
  cudaMalloc(&bias_gpu_, sizeof(float) * bias_size_);
  cudaMemcpy(bias_gpu_,
             bias_.data(),
             bias_size_ * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&scale_gpu_, sizeof(float) * scale_size_);
  cudaMemcpy(scale_gpu_,
             scale_.data(),
             scale_size_ * sizeof(float),
             cudaMemcpyHostToDevice);
  if (with_fp16_) {
    cudaMalloc(&fp16_bias_gpu_, sizeof(half) * bias_size_);
    cudaMemcpy(fp16_bias_gpu_,
               fp16_bias_.data(),
               bias_size_ * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMalloc(&fp16_scale_gpu_, sizeof(half) * scale_size_);
    cudaMemcpy(fp16_scale_gpu_,
               fp16_scale_.data(),
               scale_size_ * sizeof(half),
               cudaMemcpyHostToDevice);
  }
  if (ele_bias_size_ > 0) {
    if (with_fp16_) {
      cudaMalloc(&ele_bias_gpu_, sizeof(half) * ele_bias_size_);
      cudaMemcpy(ele_bias_gpu_,
                 fp16_ele_bias_.data(),
                 ele_bias_size_ * sizeof(half),
                 cudaMemcpyHostToDevice);
    } else {
      cudaMalloc(&ele_bias_gpu_, sizeof(float) * ele_bias_size_);
      cudaMemcpy(ele_bias_gpu_,
                 fp32_ele_bias_.data(),
                 ele_bias_size_ * sizeof(float),
                 cudaMemcpyHostToDevice);
    }
  } else {
    ele_bias_gpu_ = nullptr;
  }

  return 0;
}

void PIRPrelnResidualBiasPluginDynamic::terminate() TRT_NOEXCEPT {
  if (bias_gpu_) {
    cudaFree(bias_gpu_);
    bias_gpu_ = nullptr;
  }
  if (fp16_bias_gpu_) {
    cudaFree(fp16_bias_gpu_);
    fp16_bias_gpu_ = nullptr;
  }
  if (scale_gpu_) {
    cudaFree(scale_gpu_);
    scale_gpu_ = nullptr;
  }
  if (fp16_scale_gpu_) {
    cudaFree(fp16_scale_gpu_);
    fp16_scale_gpu_ = nullptr;
  }
  if (ele_bias_gpu_) {
    cudaFree(ele_bias_gpu_);
    ele_bias_gpu_ = nullptr;
  }
}

nvinfer1::IPluginV2DynamicExt *PIRPrelnResidualBiasPluginDynamic::clone() const
    TRT_NOEXCEPT {
  PIRPrelnResidualBiasPluginDynamic *ptr = nullptr;
  if (with_fp16_) {
    ptr = new PIRPrelnResidualBiasPluginDynamic(bias_.data(),
                                                scale_.data(),
                                                fp16_ele_bias_.data(),
                                                bias_size_,
                                                scale_size_,
                                                ele_bias_size_,
                                                eps_,
                                                with_fp16_);
  } else {
    ptr = new PIRPrelnResidualBiasPluginDynamic(bias_.data(),
                                                scale_.data(),
                                                fp32_ele_bias_.data(),
                                                bias_size_,
                                                scale_size_,
                                                ele_bias_size_,
                                                eps_,
                                                with_fp16_);
  }

  ptr->bias_gpu_ = bias_gpu_;
  ptr->fp16_bias_gpu_ = fp16_bias_gpu_;
  ptr->scale_gpu_ = scale_gpu_;
  ptr->fp16_scale_gpu_ = fp16_scale_gpu_;
  ptr->ele_bias_gpu_ = ele_bias_gpu_;
  return ptr;
}

const char *PIRPrelnResidualBiasPluginDynamic::getPluginType() const
    TRT_NOEXCEPT {
  return "preln_residual_bias_plugin_dynamic";
}

int PIRPrelnResidualBiasPluginDynamic::getNbOutputs() const TRT_NOEXCEPT {
  return 2;
}

size_t PIRPrelnResidualBiasPluginDynamic::getSerializationSize() const
    TRT_NOEXCEPT {
  size_t ser_size = SerializedSize(bias_) + SerializedSize(fp16_bias_) +
                    SerializedSize(scale_) + SerializedSize(fp16_scale_) +
                    SerializedSize(fp32_ele_bias_) +
                    SerializedSize(fp16_ele_bias_) +
                    SerializedSize(bias_size_) + SerializedSize(scale_size_) +
                    SerializedSize(ele_bias_size_) + SerializedSize(eps_) +
                    SerializedSize(with_fp16_);
  return ser_size;
}
void PIRPrelnResidualBiasPluginDynamic::serialize(void *buffer) const
    TRT_NOEXCEPT {
  SerializeValue(&buffer, bias_);
  SerializeValue(&buffer, fp16_bias_);
  SerializeValue(&buffer, scale_);
  SerializeValue(&buffer, fp16_scale_);
  SerializeValue(&buffer, fp32_ele_bias_);
  SerializeValue(&buffer, fp16_ele_bias_);
  SerializeValue(&buffer, bias_size_);
  SerializeValue(&buffer, scale_size_);
  SerializeValue(&buffer, ele_bias_size_);
  SerializeValue(&buffer, eps_);
  SerializeValue(&buffer, with_fp16_);
}

nvinfer1::DimsExprs PIRPrelnResidualBiasPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  if (output_index < 2) {
    return inputs[0];
  } else {  // moving mean and var
    nvinfer1::DimsExprs ret;
    ret.nbDims = 1;
    ret.d[0] = inputs[0].d[2];
    return ret;
  }
}

bool PIRPrelnResidualBiasPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of swish plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVAILABLE
      return (in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
#else
      PADDLE_THROW(
          common::errors::Fatal("TRT plugin supported FP16 is not available "
                                "while with_fp16 is set true."));
#endif
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];

  if (pos == 1) {
    return in.type == prev.type && in.format == prev.format;
  }

  // output
  return in.type == prev.type && in.format == prev.format;
}

void PIRPrelnResidualBiasPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nb_outputs) TRT_NOEXCEPT {}

size_t PIRPrelnResidualBiasPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs,
    int nb_inputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nb_outputs) const TRT_NOEXCEPT {
  return 0;
}

nvinfer1::DataType PIRPrelnResidualBiasPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

void PIRPrelnResidualBiasPluginDynamic::destroy() TRT_NOEXCEPT { delete this; }

int PIRPrelnResidualBiasPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  int hidden;
  int rows_temp;
  int cols_temp;
  if (input_dims.nbDims == 3) {
    hidden = input_dims.d[2];
    rows_temp = input_dims.d[0] * input_dims.d[1];
    cols_temp = input_dims.d[2];
  } else if (input_dims.nbDims == 4) {
    hidden = input_dims.d[3];
    rows_temp = input_dims.d[0] * input_dims.d[1] * input_dims.d[2];
    cols_temp = input_dims.d[3];
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "fused_bias_dropout_residual_layer_norm op only support 3-D or 4-D "
        "input, but get `%d`-D.",
        input_dims.nbDims));
  }
  const size_t rows = static_cast<size_t>(rows_temp);  // batch * seq_length
  const size_t cols = static_cast<size_t>(cols_temp);

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. PrelnResidualBias-->fp32";
    const float *input1 = static_cast<const float *>(inputs[0]);
    const float *input2 = static_cast<const float *>(inputs[1]);

    uint64_t seed = 0;
    const float dropout_prob = 0.;
    const bool is_upscale_in_train = false;
    const bool is_test = true;
    const uint64_t increment = 0;
    const float epsilon = eps_;
    const float *src = input2;
    const float *residual = input1;
    const float *bias = static_cast<float *>(ele_bias_gpu_);
    const float *scale = scale_gpu_;
    const float *layernorm_bias = bias_gpu_;
    uint8_t *mask_data = nullptr;
    float *dst = static_cast<float *>(outputs[1]);
    float *layernorm_dst = static_cast<float *>(outputs[0]);
    float *mean = nullptr;
    float *var = nullptr;
    const int VecSize = 8;
    phi::fusion::FusedLayernormResidualDropoutBiasFunctor<float,
                                                          uint8_t,
                                                          VecSize,
                                                          float,
                                                          false>()(
        rows,
        cols,
        seed,
        dropout_prob,
        is_upscale_in_train,
        is_test,
        increment,
        epsilon,
        src,
        residual,
        bias,
        scale,
        layernorm_bias,
        mask_data,
        dst,
        layernorm_dst,
        mean,
        var,
        stream);

  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef TRT_PLUGIN_FP16_AVAILABLE
    VLOG(1) << "TRT Plugin DataType selected. PrelnResidualBias-->fp16";
    const half *input1 = static_cast<const half *>(inputs[0]);
    const half *input2 = static_cast<const half *>(inputs[1]);

    uint64_t seed = 0;
    const float dropout_prob = 0.;
    const bool is_upscale_in_train = false;
    const bool is_test = true;
    const uint64_t increment = 0;
    const float epsilon = eps_;
    const half *src = input2;
    const half *residual = input1;
    const half *bias = static_cast<half *>(ele_bias_gpu_);
    const float *scale = scale_gpu_;
    const float *layernorm_bias = bias_gpu_;
    uint8_t *mask_data = nullptr;
    half *dst = static_cast<half *>(outputs[1]);
    half *layernorm_dst = static_cast<half *>(outputs[0]);
    float *mean = nullptr;
    float *var = nullptr;
    const int VecSize = 8;

    int sm = getSMVersion();
    // sm >= 60 to support _ldg
    if (sm >= 60) {
      // if hidden is even, use half2 kernel generalAddBiasResidualLayerNormOpt2
      if (hidden % 2 == 0) {
        int half_n = hidden / 2;
        int half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(std::min(half_n_32, 512));
        int rolls_per_thread = half_n / block.x;
        int unroll_factor = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
          unroll_factor /= 2;
        }
        switch (unroll_factor) {
          case 1:
            HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(1);
            break;
          case 2:
            HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(2);
            break;
          case 4:
            HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(4);
            break;
          case 8:
            HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(8);
            break;
          default:
            PADDLE_THROW(common::errors::Fatal(
                "Invalid UNROLL_FACTOR in preln_residual_bias trt plugin."));
        }
      } else {
        phi::fusion::FusedLayernormResidualDropoutBiasFunctor<half,
                                                              uint8_t,
                                                              VecSize,
                                                              float,
                                                              false>()(
            rows,
            cols,
            seed,
            dropout_prob,
            is_upscale_in_train,
            is_test,
            increment,
            epsilon,
            src,
            residual,
            bias,
            scale,
            layernorm_bias,
            mask_data,
            dst,
            layernorm_dst,
            mean,
            var,
            stream);
      }
    } else {
      // if sm < 60, use FusedLayernormResidualDropoutBiasFunctor only
      phi::fusion::FusedLayernormResidualDropoutBiasFunctor<half,
                                                            uint8_t,
                                                            VecSize,
                                                            float,
                                                            false>()(
          rows,
          cols,
          seed,
          dropout_prob,
          is_upscale_in_train,
          is_test,
          increment,
          epsilon,
          src,
          residual,
          bias,
          scale,
          layernorm_bias,
          mask_data,
          dst,
          layernorm_dst,
          mean,
          var,
          stream);
    }
#else
    PADDLE_THROW(common::errors::Fatal(
        "The Ernie(Bert) tensorRT plugin should be "
        "complied with CUDA version >= 10.0 when running with fp16. "
        "Please recompile it or try to use fp32 by set "
        "config.SetTRTDynamicShapeInfo(min_input_shape, "
        "max_input_shape, opt_input_shape, true"));
#endif
  } else {
    PADDLE_THROW(
        common::errors::Fatal("The PrelnResidualBias TRT Plugin's input type "
                              "should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

const char *PIRPrelnResidualBiasPluginDynamicCreator::getPluginName() const
    TRT_NOEXCEPT {
  return "pir_preln_residual_bias_plugin_dynamic";
}

const char *PIRPrelnResidualBiasPluginDynamicCreator::getPluginVersion() const
    TRT_NOEXCEPT {
  return "1";
}

nvinfer1::IPluginV2 *
PIRPrelnResidualBiasPluginDynamicCreator::deserializePlugin(
    const char *name,
    const void *serial_data,
    size_t serial_length) TRT_NOEXCEPT {
  return new PIRPrelnResidualBiasPluginDynamic(serial_data, serial_length);
}

nvinfer1::IPluginV2 *PIRPrelnResidualBiasPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  const float *bias = nullptr;
  const float *scale = nullptr;
  const float *ele_bias = nullptr;
  const half *ele_bias_fp16 = nullptr;
  int bias_size = 0;
  int scale_size = 0;
  int ele_bias_size = 0;
  float epsilon = 0.0f;
  bool with_fp16 = false;

  for (int i = 0; i < fc->nbFields; ++i) {
    const nvinfer1::PluginField &f = fc->fields[i];
    const std::string field_name(f.name);
    if (field_name.compare("with_fp16") == 0) {
      with_fp16 = *static_cast<const bool *>(f.data);
    }
  }

  for (int i = 0; i < fc->nbFields; ++i) {
    const nvinfer1::PluginField &f = fc->fields[i];
    const std::string field_name(f.name);

    if (field_name.compare("bias") == 0) {
      bias = static_cast<const float *>(f.data);
    } else if (field_name.compare("scale") == 0) {
      scale = static_cast<const float *>(f.data);
    } else if (field_name.compare("ele_bias") == 0) {
      if (with_fp16) {
        ele_bias_fp16 = static_cast<const half *>(f.data);
      } else {
        ele_bias = static_cast<const float *>(f.data);
      }
    } else if (field_name.compare("bias_size") == 0) {
      bias_size = *static_cast<const int *>(f.data);
    } else if (field_name.compare("scale_size") == 0) {
      scale_size = *static_cast<const int *>(f.data);
    } else if (field_name.compare("ele_bias_size") == 0) {
      ele_bias_size = *static_cast<const int *>(f.data);
    } else if (field_name.compare("epsilon") == 0) {
      epsilon = *static_cast<const float *>(f.data);
    } else {
      assert(false && "unknown plugin field name.");
    }
  }

  if (with_fp16) {
    return new PIRPrelnResidualBiasPluginDynamic(bias,
                                                 scale,
                                                 ele_bias_fp16,
                                                 bias_size,
                                                 scale_size,
                                                 ele_bias_size,
                                                 epsilon,
                                                 with_fp16);
  } else {
    return new PIRPrelnResidualBiasPluginDynamic(bias,
                                                 scale,
                                                 ele_bias,
                                                 bias_size,
                                                 scale_size,
                                                 ele_bias_size,
                                                 epsilon,
                                                 with_fp16);
  }
}

#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
